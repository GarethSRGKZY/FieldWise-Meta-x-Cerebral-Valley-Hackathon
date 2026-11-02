"""The Fieldwise pipeline: one place where Glimmer, Spark and SAM 3.1 meet.

ask():       question -> guides (BM25) -> Glimmer -> confidence -> route
diagnose():  photo -> strip EXIF/GPS -> Glimmer picks a class -> route
severity():  photo -> SAM 3.1 leaf + lesion masks -> % of leaf affected

The same functions are used by the app and by the benchmark, so the numbers
in the write-up come from the code the judges see.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from . import redact
from .config import settings
from .confidence import choice_from_logprobs, extract_answer_text
from .connectivity import is_online
from .imageprep import prepare
from .labels import LABELS, UNSURE_LETTER, options_for
from .prompts import DX_SYSTEM, QA_SYSTEM, dx_user_prompt, qa_user_prompt
from .providers.glimmer import GlimmerClient, GlimmerUnavailable
from .providers.sam import SamClient, SamUnavailable
from .providers.spark import SparkBlocked, SparkClient, SparkDisabled, SparkUnavailable
from .queue import EscalationQueue
from .replay import GlimmerReplay, describe
from .retrieval import GuideIndex, Passage
from .router import Route, decide, dx_confidence, qa_confidence
from .safety import PPE_REMINDER, REFERRAL, SafetyReport, check


@dataclass
class ModelOutput:
    model: str
    raw: str
    latency_s: float
    cost_usd: float = 0.0
    cached: bool = False
    confidence_source: str = ""


@dataclass
class AskResult:
    answer: str
    supported: bool
    confidence: float | None    # None = Glimmer didn't run, so there is no on-device confidence
    route: str
    answered_by: str
    sources: list[dict]
    redactions: dict
    safety_flags: list[str]
    local: dict
    cloud: dict | None = None
    queue_id: str | None = None
    latency_s: float = 0.0
    cost_usd: float = 0.0
    coverage: float = 0.0
    note: str = ""
    local_available: bool = True
    replayed: dict | None = None    # set when Glimmer's output is a replay of the benchmark run


@dataclass
class DiagnoseResult:
    label: str | None           # e.g. "maize/fall_armyworm"; None = unsure
    label_name: str
    confidence: float | None
    route: str
    answered_by: str
    probabilities: dict
    photo: dict
    redactions: dict
    local: dict
    cloud: dict | None = None
    queue_id: str | None = None
    latency_s: float = 0.0
    cost_usd: float = 0.0
    note: str = ""
    local_available: bool = True
    replayed: dict | None = None


GLIMMER_DOWN = "The hub model (Muse Glimmer) isn't running."
SPARK_DECLINED = "Muse Spark's content filter declined this question, so it won't be sent again."
SPARK_OFF = "Cloud second opinions (Muse Spark) are switched off on this hub"
# Shown instead of an answer when no model ran. REFERRAL means "the guides don't cover this", which would be untrue here.
NO_REPLAY = ("Demo replay mode: Muse Glimmer has no recorded answer for this, and this hub is too small "
             "to run it live.")
NO_ANSWER_YET = ("No answer yet. It will appear in the queue below once Muse Spark replies. "
                 "If the problem is urgent, contact your local agricultural extension officer.")


class Fieldwise:
    def __init__(self, index: GuideIndex | None = None, glimmer: GlimmerClient | None = None,
                 spark: SparkClient | None = None, sam: SamClient | None = None,
                 queue: EscalationQueue | None = None, replay: GlimmerReplay | None = None):
        self.index = index or GuideIndex.load(settings.index_path)
        self.glimmer = glimmer or GlimmerClient()
        self.spark = spark or SparkClient()
        self.sam = sam or SamClient()
        self.queue = queue or EscalationQueue(settings.queue_db)
        self.replay = replay if replay is not None else (GlimmerReplay() if settings.glimmer_replay else None)

    def glimmer_health(self) -> tuple[bool, str]:
        return self.replay.health_detail() if self.replay else self.glimmer.health_detail()

    # ------------------------------------------------------------------ Q&A
    def _parse_qa(self, raw: str, logprobs: list[dict] | None, passages: list[Passage], coverage: float, finish: str | None):
        choice = choice_from_logprobs(logprobs, "SUPPORTED:", ["YES", "NO"], raw)
        answer = extract_answer_text(raw)
        complete = bool(answer) and finish != "length"
        report = check(answer, passages) if answer else None
        supported = choice.chosen == "YES"
        cited_ok = (not supported) or bool(report and report.cited)
        conf = qa_confidence(choice.p_chosen, coverage, cited_ok, complete)
        return answer, supported, conf, choice, report

    def local_qa(self, question: str, crop: str | None = None) -> dict:
        res = self.index.search(f"{crop or ''} {question}", k=settings.retrieval_k)
        out = self.glimmer.chat(QA_SYSTEM, qa_user_prompt(question, res.passages, crop))
        answer, supported, conf, choice, report = self._parse_qa(out.content, out.logprobs, res.passages, res.coverage, out.finish_reason)
        return {
            "answer": answer, "supported": supported, "confidence": conf,
            "p_supported_choice": choice.p_chosen, "confidence_source": choice.source,
            "coverage": res.coverage, "passages": res.passages, "report": report,
            "raw": out.content, "reasoning_chars": len(out.reasoning),
            "latency_s": out.latency_s, "finish_reason": out.finish_reason, "usage": out.usage,
        }

    def _replay_qa(self, question: str, crop: str | None) -> dict:
        """Glimmer's recorded benchmark answer for this question (see fieldwise/replay.py). The guide search
        runs live with the recorded question, so the sources shown are the passages Glimmer was given."""
        found = self.replay.qa(question, crop)
        if found is None:
            raise GlimmerUnavailable("replay mode: no recorded Glimmer answer for this question and crop", NO_REPLAY)
        rec = found.rec
        res = self.index.search(f"{crop or ''} {found.item['question']}", k=settings.retrieval_k)
        shown = rec.get("answer_shown") or ""
        report = SafetyReport(shown, rec.get("removed_sentences") or [], PPE_REMINDER in shown, rec.get("cited") or [])
        return {
            "answer": shown, "supported": bool(rec.get("supported")), "confidence": float(rec["confidence"]),
            "p_supported_choice": rec.get("p_supported_choice"), "confidence_source": rec.get("confidence_source"),
            "coverage": res.coverage, "passages": res.passages, "report": report, "raw": "", "reasoning_chars": 0,
            "latency_s": rec.get("latency_s"), "finish_reason": rec.get("finish_reason"), "usage": rec.get("usage"),
            "replayed": describe(rec) | {"benchmark_id": found.id},
        }

    def cloud_qa(self, redacted_question: str, passages: list[Passage], coverage: float, crop: str | None = None) -> dict:
        out = self.spark.chat(QA_SYSTEM, qa_user_prompt(redacted_question, passages, crop))
        answer, supported, _, _, report = self._parse_qa(out.content, None, passages, coverage, None)
        return {"answer": answer, "supported": supported, "report": report, "raw": out.content,
                "latency_s": out.latency_s, "cost_usd": out.cost_usd, "cached": out.cached,
                "usage": out.usage, "model": out.model or settings.spark_model}

    def ask(self, question: str, crop: str | None = None, threshold: float | None = None) -> AskResult:
        t0 = time.perf_counter()
        threshold = settings.qa_threshold if threshold is None else threshold
        redacted_q, spans = redact.redact_text(question)

        try:
            local = self._replay_qa(question, crop) if self.replay else self.local_qa(question, crop)
        except GlimmerUnavailable as e:
            return self._ask_without_glimmer(question, redacted_q, crop, redact.summary(spans), str(e), t0, e.short)

        online = is_online()
        route = decide(local["confidence"], threshold, online)
        final, cloud_info, queue_id, cost = local, None, None, 0.0
        answered_by = "Muse Glimmer (on-device)" + (", replayed from the benchmark run" if local.get("replayed") else "")

        if route == Route.CLOUD:
            try:
                cloud = self.cloud_qa(redacted_q, local["passages"], local["coverage"], crop)
                final, cost, answered_by = cloud, cloud["cost_usd"], f"Muse Spark ({cloud['model']}, cloud)"
                cloud_info = {k: v for k, v in cloud.items() if k not in ("report",)}
            except SparkUnavailable as e:
                route = Route.LOCAL_FALLBACK
                cloud_info = {"error": str(e), "blocked": isinstance(e, SparkBlocked), "disabled": isinstance(e, SparkDisabled)}
        blocked = bool(cloud_info and cloud_info.get("blocked"))
        if route in (Route.QUEUED, Route.LOCAL_FALLBACK) and not blocked:
            queue_id = self._queue_question(redacted_q, crop, local["passages"], local["answer"])

        report = final["report"]
        answer_text = report.text if report else ""
        supported = final["supported"]
        if not supported or not answer_text:
            answer_text = (answer_text + "\n\n" if answer_text else "") + REFERRAL
        if blocked:
            answer_text = ("Tentative answer (low confidence on the hub, and no second opinion is available; "
                           "please check with your extension officer):\n\n" + answer_text)
        elif route in (Route.QUEUED, Route.LOCAL_FALLBACK):
            answer_text = "Tentative answer (low confidence; a second opinion is queued):\n\n" + answer_text

        cited = set(report.cited) if report else set()
        sources = [p.cite() | {"n": i} for i, p in enumerate(local["passages"], start=1) if i in cited]
        local_view = {k: v for k, v in local.items() if k not in ("passages", "report")}

        return AskResult(
            answer=answer_text, supported=supported, confidence=round(local["confidence"], 4),
            route=route.value, answered_by=answered_by, sources=sources,
            redactions=redact.summary(spans), safety_flags=report.flags if report else [],
            local=local_view, cloud=cloud_info, queue_id=queue_id,
            latency_s=time.perf_counter() - t0, cost_usd=cost, coverage=round(local["coverage"], 3),
            note=(f"Low confidence on the hub. {SPARK_DECLINED} This is Muse Glimmer's tentative answer." if blocked
                  else f"Low confidence on the hub. {SPARK_OFF}, so this is Muse Glimmer's tentative answer. "
                       "A redacted copy is queued for when they're switched on."
                  if cloud_info and cloud_info.get("disabled") else ""),
            replayed=local.get("replayed"),
        )

    def _queue_question(self, redacted_q: str, crop: str | None, passages: list[Passage], local_answer: str | None) -> str:
        return self.queue.add("ask", {"question": redacted_q, "crop": crop,
                                      "chunk_ids": [p.chunk_id for p in passages], "local_answer": local_answer})

    def _ask_without_glimmer(self, question: str, redacted_q: str, crop: str | None, red: dict,
                             err: str, t0: float, why: str = GLIMMER_DOWN) -> AskResult:
        """Glimmer is down. Online: send the redacted question straight to Spark. Offline: queue it.
        Never present a fixed referral as if a model had written it."""
        res = self.index.search(f"{crop or ''} {question}", k=settings.retrieval_k)   # retrieval needs no model
        common = dict(redactions=red, local={"error": err}, coverage=round(res.coverage, 3), local_available=False)
        if not is_online():
            queue_id = self._queue_question(redacted_q, crop, res.passages, None)
            return AskResult(NO_ANSWER_YET, False, None, Route.QUEUED.value, "none", [], safety_flags=[], queue_id=queue_id,
                             latency_s=time.perf_counter() - t0,
                             note=f"{why} There's no connection either, so a redacted copy is queued for Muse Spark.",
                             **common)
        try:
            cloud = self.cloud_qa(redacted_q, res.passages, res.coverage, crop)
        except SparkBlocked as e:
            return AskResult(REFERRAL, False, None, Route.UNAVAILABLE.value, "none", [], safety_flags=[],
                             cloud={"error": str(e), "blocked": True}, latency_s=time.perf_counter() - t0,
                             note=f"{why} {SPARK_DECLINED}", **common)
        except SparkUnavailable as e:
            return AskResult(NO_ANSWER_YET, False, None, Route.UNAVAILABLE.value, "none", [], safety_flags=[],
                             cloud={"error": str(e)}, queue_id=self._queue_question(redacted_q, crop, res.passages, None),
                             latency_s=time.perf_counter() - t0,
                             note=(f"{why} {SPARK_OFF} too, so no model has answered yet. "
                                   "A redacted copy is queued." if isinstance(e, SparkDisabled) else
                                   f"{why} Muse Spark couldn't be reached either, so no model has answered yet. "
                                   "A redacted copy is queued for Muse Spark."), **common)
        rep = cloud["report"]
        text = rep.text if rep else ""
        if not cloud["supported"] or not text:
            text = (text + "\n\n" if text else "") + REFERRAL
        cited = set(rep.cited) if rep else set()
        sources = [p.cite() | {"n": i} for i, p in enumerate(res.passages, start=1) if i in cited]
        return AskResult(text, cloud["supported"], None, Route.CLOUD.value, f"Muse Spark ({cloud['model']}, cloud)",
                         sources, safety_flags=rep.flags if rep else [],
                         cloud={k: v for k, v in cloud.items() if k != "report"}, cost_usd=cloud["cost_usd"],
                         latency_s=time.perf_counter() - t0,
                         note=f"{why} A redacted copy went straight to Muse Spark.", **common)

    # ------------------------------------------------------------ diagnosis
    def _parse_dx(self, raw: str, logprobs, crop: str, finish: str | None):
        opts = options_for(crop)
        letters = [l for l, _ in opts]
        choice = choice_from_logprobs(logprobs, "ANSWER:", letters, raw)
        complete = choice.chosen is not None and finish != "length"
        unsure = choice.chosen == UNSURE_LETTER
        conf = dx_confidence(choice.p_chosen, unsure, complete)
        lab = dict(opts).get(choice.chosen) if choice.chosen else None
        probs = {(dict(opts)[l].key if dict(opts).get(l) else "unsure"): round(p, 4) for l, p in choice.probs.items() if l in dict(opts)}
        return lab, conf, probs, choice

    def local_dx(self, jpeg: bytes, crop: str, note: str | None = None) -> dict:
        out = self.glimmer.chat(DX_SYSTEM, dx_user_prompt(crop, note), image_jpeg=jpeg)
        lab, conf, probs, choice = self._parse_dx(out.content, out.logprobs, crop, out.finish_reason)
        return {"label": lab.key if lab else None, "confidence": conf, "probabilities": probs,
                "confidence_source": choice.source, "raw": out.content, "latency_s": out.latency_s,
                "finish_reason": out.finish_reason, "usage": out.usage}

    def _replay_dx(self, image_bytes: bytes, crop: str) -> dict:
        found = self.replay.dx(image_bytes, crop)
        if found is None:
            raise GlimmerUnavailable("replay mode: no recorded Glimmer diagnosis for this photo and crop",
                                     NO_REPLAY.replace("for this", "for this photo"))
        rec = found.rec
        return {"label": rec.get("pred"), "confidence": float(rec["confidence"]), "probabilities": rec.get("probabilities") or {},
                "confidence_source": rec.get("confidence_source"), "raw": rec.get("raw") or "", "latency_s": rec.get("latency_s"),
                "finish_reason": rec.get("finish_reason"), "usage": rec.get("usage"),
                "replayed": describe(rec) | {"benchmark_id": found.id}}

    def cloud_dx(self, jpeg: bytes, crop: str, redacted_note: str | None = None) -> dict:
        out = self.spark.chat(DX_SYSTEM, dx_user_prompt(crop, redacted_note), image_jpeg=jpeg)
        lab, _, _, _ = self._parse_dx(out.content, None, crop, None)
        return {"label": lab.key if lab else None, "raw": out.content, "latency_s": out.latency_s,
                "cost_usd": out.cost_usd, "cached": out.cached, "usage": out.usage,
                "model": out.model or settings.spark_model}

    def diagnose(self, image_bytes: bytes, crop: str, note: str | None = None, threshold: float | None = None) -> DiagnoseResult:
        t0 = time.perf_counter()
        threshold = settings.dx_threshold if threshold is None else threshold
        photo = prepare(image_bytes)
        redacted_note, spans = redact.redact_text(note or "")
        photo_info = {"width": photo.width, "height": photo.height,
                      "exif_removed": photo.had_exif, "gps_removed": photo.had_gps}
        red = redact.summary(spans) | ({"PHOTO_GPS": 1} if photo.had_gps else {})

        try:
            local = self._replay_dx(image_bytes, crop) if self.replay else self.local_dx(photo.jpeg, crop, note)
        except GlimmerUnavailable as e:
            return self._diagnose_without_glimmer(photo.jpeg, crop, redacted_note, photo_info, red, str(e), t0, e.short)

        online = is_online()
        route = decide(local["confidence"], threshold, online)
        label_key, cloud_info, queue_id, cost = local["label"], None, None, 0.0
        answered_by = "Muse Glimmer (on-device)" + (", replayed from the benchmark run" if local.get("replayed") else "")

        if route == Route.CLOUD:
            try:
                cloud = self.cloud_dx(photo.jpeg, crop, redacted_note or None)
                label_key, cost = cloud["label"], cloud["cost_usd"]
                answered_by = f"Muse Spark ({cloud['model']}, cloud)"
                cloud_info = cloud
            except SparkUnavailable as e:
                route, cloud_info = Route.LOCAL_FALLBACK, {"error": str(e), "blocked": isinstance(e, SparkBlocked),
                                                           "disabled": isinstance(e, SparkDisabled)}
        blocked = bool(cloud_info and cloud_info.get("blocked"))
        if route in (Route.QUEUED, Route.LOCAL_FALLBACK) and not blocked:
            queue_id = self._queue_photo(photo.jpeg, crop, redacted_note, local["label"])

        lab = LABELS.get(label_key) if label_key else None
        name = lab.name if lab else "Unsure: please show the plant to your extension officer"
        return DiagnoseResult(
            label=label_key, label_name=name, confidence=round(local["confidence"], 4), route=route.value,
            answered_by=answered_by, probabilities=local["probabilities"], photo=photo_info,
            redactions=red, local=local, cloud=cloud_info, queue_id=queue_id,
            latency_s=time.perf_counter() - t0, cost_usd=cost,
            note=(f"Low confidence on the hub. {SPARK_DECLINED.replace('question', 'photo')} "
                  "This is Muse Glimmer's tentative result." if blocked
                  else f"Low confidence on the hub. {SPARK_OFF}, so this is Muse Glimmer's tentative result. "
                       "The photo is queued for when they're switched on."
                  if cloud_info and cloud_info.get("disabled") else ""),
            replayed=local.get("replayed"),
        )

    def _queue_photo(self, jpeg: bytes, crop: str, redacted_note: str, local_label: str | None) -> str:
        settings.outbox_dir.mkdir(parents=True, exist_ok=True)
        img_path = settings.outbox_dir / f"{int(time.time() * 1000)}.jpg"
        img_path.write_bytes(jpeg)   # EXIF-free copy only
        return self.queue.add("diagnose", {"crop": crop, "note": redacted_note, "image_path": str(img_path),
                                           "local_label": local_label})

    def _diagnose_without_glimmer(self, jpeg: bytes, crop: str, redacted_note: str, photo_info: dict, red: dict,
                                  err: str, t0: float, why: str = GLIMMER_DOWN) -> DiagnoseResult:
        common = dict(photo=photo_info, redactions=red, local={"error": err}, local_available=False)
        if not is_online():
            return DiagnoseResult(None, "No result yet: waiting for a second opinion", None, Route.QUEUED.value, "none",
                                  {}, queue_id=self._queue_photo(jpeg, crop, redacted_note, None),
                                  latency_s=time.perf_counter() - t0,
                                  note=f"{why} There's no connection either, so the photo is queued for Muse Spark.",
                                  **common)
        try:
            cloud = self.cloud_dx(jpeg, crop, redacted_note or None)
        except SparkBlocked as e:
            return DiagnoseResult(None, "Unsure: please show the plant to your extension officer", None,
                                  Route.UNAVAILABLE.value, "none", {}, cloud={"error": str(e), "blocked": True},
                                  latency_s=time.perf_counter() - t0,
                                  note=f"{why} {SPARK_DECLINED.replace('question', 'photo')}", **common)
        except SparkUnavailable as e:
            return DiagnoseResult(None, "No result yet: no model could answer", None, Route.UNAVAILABLE.value, "none",
                                  {}, cloud={"error": str(e)}, queue_id=self._queue_photo(jpeg, crop, redacted_note, None),
                                  latency_s=time.perf_counter() - t0,
                                  note=(f"{why} {SPARK_OFF} too, so the photo is queued." if isinstance(e, SparkDisabled)
                                        else f"{why} Muse Spark couldn't be reached either, so the photo is queued "
                                             "for Muse Spark."), **common)
        lab = LABELS.get(cloud["label"]) if cloud["label"] else None
        return DiagnoseResult(cloud["label"], lab.name if lab else "Unsure: please show the plant to your extension officer",
                              None, Route.CLOUD.value, f"Muse Spark ({cloud['model']}, cloud)", {}, cloud=cloud,
                              cost_usd=cloud["cost_usd"], latency_s=time.perf_counter() - t0,
                              note=f"{why} The photo went straight to Muse Spark.", **common)

    # ------------------------------------------------------------- severity
    def severity(self, image_bytes: bytes, label_key: str | None) -> dict:
        photo = prepare(image_bytes)
        lab = LABELS.get(label_key) if label_key else None
        phrase = lab.sam_phrase if lab else "leaf spot"
        if lab and lab.kind == "pest":
            return {"available": False, "error": "not_validated",
                    "note": "Damage measurement is only checked for leaf diseases, not insect damage, so it isn't shown here."}
        if not is_online():
            return {"available": False, "error": "offline",
                    "note": "Measuring the damaged area needs a connection (SAM 3.1 runs on the Meta Model API). Try again when online."}
        try:
            leaf_phrase = settings.sam_leaf_phrase.replace("{crop}", lab.crop) if lab else "leaf"
            res = self.sam.severity(photo.jpeg, phrase, leaf_phrase=leaf_phrase)
        except SamUnavailable as e:
            return {"available": False, "error": str(e),
                    "note": "Severity needs a connection (SAM 3.1 runs on the Meta Model API)."}
        return {"available": True, **asdict(res), "show_percent": settings.severity_show_percent,
                "error_pts": settings.severity_error_pts}

    # -------------------------------------------------------- queue flushing
    def flush_queue(self) -> list[dict]:
        if not is_online():
            return []
        done = []
        for item in self.queue.pending():
            p = item["payload"]
            try:
                if item["kind"] == "ask":
                    by_id = {c["chunk_id"]: c for c in self.index.chunks}
                    passages = [Passage(c["chunk_id"], c["doc_id"], c["title"], c.get("page"), c["text"],
                                        c.get("url", ""), c.get("licence", "")) for cid in p["chunk_ids"] if (c := by_id.get(cid))]
                    cloud = self.cloud_qa(p["question"], passages, 1.0, p.get("crop"))
                    rep = cloud["report"]
                    text = rep.text if rep else ""
                    if not cloud["supported"] or not text:
                        text = (text + "\n\n" if text else "") + REFERRAL
                    result = {"answer": text, "answered_by": f"Muse Spark ({cloud['model']}, cloud)",
                              "cost_usd": cloud["cost_usd"], "flags": rep.flags if rep else []}
                else:
                    jpeg = open(p["image_path"], "rb").read()
                    cloud = self.cloud_dx(jpeg, p["crop"], p.get("note") or None)
                    lab = LABELS.get(cloud["label"]) if cloud["label"] else None
                    result = {"label": cloud["label"], "label_name": lab.name if lab else "Unsure",
                              "answered_by": f"Muse Spark ({cloud['model']}, cloud)", "cost_usd": cloud["cost_usd"]}
                self.queue.mark(item["id"], "done", result)
                done.append({"id": item["id"], **result})
            except SparkBlocked as e:
                self.queue.mark(item["id"], "blocked", {"error": str(e)})
                done.append({"id": item["id"], "error": "Muse Spark's content filter declined it; ask your extension officer."})
            except (SparkUnavailable, OSError) as e:
                self.queue.bump(item["id"])
                done.append({"id": item["id"], "error": str(e)})
        return done
