"""Run the benchmark: every item through Glimmer (local) and Spark (cloud).

    python bench/run_bench.py --task images --setup local --split all
    python bench/run_bench.py --task images --setup cloud --split all
    python bench/run_bench.py --task qa     --setup both  --split all

Each setup runs on every item independently. The routed setup is then
computed per item by bench/analyze.py: Glimmer's answer where its confidence
clears the threshold, Spark's answer otherwise. That lets us sweep thresholds
without paying for extra runs. Confirm with a small live run through the app
and say in the write-up that routed numbers were replayed this way.

The runner is resumable: finished item ids are skipped, so an overnight run
can be restarted after a crash or a laptop sleep.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fieldwise.config import settings  # noqa: E402
from fieldwise.imageprep import prepare  # noqa: E402
from fieldwise.pipeline import Fieldwise  # noqa: E402
from fieldwise.providers.glimmer import GlimmerUnavailable  # noqa: E402
from fieldwise.providers.spark import SparkAccountBlocked, SparkBlocked, SparkUnavailable  # noqa: E402
from fieldwise.safety import PPE_REMINDER, REFERRAL  # noqa: E402
from fieldwise.safety import check  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


# --------------------------------------------------------------- inputs
def load_images(manifest: Path, split: str, image_root: Path) -> list[dict]:
    """Read the frozen manifest, resolve paths under image_root and check every file is the one frozen."""
    if not manifest.exists():
        sys.exit(f"No manifest at {manifest}. Run bench/prepare_images.py first.")
    with manifest.open() as fh:
        rows = [r for r in csv.DictReader(fh) if split == "all" or r["split"] == split]
    missing, changed = [], []
    for r in rows:
        p = Path(r["path"])
        p = p if p.is_absolute() else image_root / p
        if not p.exists():
            missing.append(str(p))
            continue
        if r.get("sha1") and hashlib.sha1(p.read_bytes()).hexdigest() != r["sha1"]:
            changed.append(str(p))
        r["path"] = str(p)
    if missing or changed:
        for what, lst in (("missing", missing), ("different from the frozen copy", changed)):
            if lst:
                print(f"{len(lst)} images {what}, e.g. {lst[0]}")
        sys.exit(f"Check --image-root (now {image_root}) points at the same CCMT copy the manifest was made from.")
    return rows


def validate_questions(rows: list[dict]) -> list[str]:
    problems = []
    ids = [r.get("id") for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("Duplicate question ids.")
    for r in rows:
        for k in ("id", "split", "question", "answerable"):
            if k not in r:
                problems.append(f"{r.get('id')}: missing '{k}'")
        if r.get("split") not in ("dev", "test"):
            problems.append(f"{r.get('id')}: split must be dev or test")
        if r.get("answerable") and not r.get("key_facts"):
            problems.append(f"{r.get('id')}: answerable questions need key_facts")
        if r.get("answerable") and not r.get("source"):
            problems.append(f"{r.get('id')}: record which guide page answers it (source)")
        if str(r.get("id", "")).startswith("TEMPLATE"):
            problems.append(f"{r.get('id')}: template row left in the question set")
    drafts = [r["id"] for r in rows if r.get("review_status") == "draft"]
    if drafts:
        problems.append(f"{len(drafts)} questions still marked review_status=draft. A person must check each "
                        "question, key fact and page, then set review_status to 'reviewed'.")
    if len(rows) < 100:
        problems.append(f"Only {len(rows)} questions; the plan says 100+.")
    return problems


def load_questions(path: Path, split: str, strict: bool) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    problems = validate_questions(rows)
    if problems:
        print("Question set problems:\n  " + "\n  ".join(problems[:30]))
        if strict:
            sys.exit("Fix these before a final run (or pass --no-strict for a smoke test).")
    return [r for r in rows if split == "all" or r["split"] == split]


# --------------------------------------------------------------- grading
def strip_boilerplate(text: str) -> str:
    """Remove the fixed texts the app adds itself (PPE reminder, referral). They contain words such as
    'spray', 'eat', 'children', 'smoke' and 'label' that are key facts for some questions, so grading
    must only look at what the model wrote."""
    for fixed in (PPE_REMINDER, REFERRAL):
        text = text.replace(fixed, " ")
    return text


def grade_qa(item: dict, supported: bool, shown_text: str) -> dict:
    """Automatic first-pass grade. Final numbers should use manual review.

    - Out-of-scope question: correct only if the answer declines (SUPPORTED: NO -> referral).
    - Answerable question: correct if supported and every key fact is present in the model's own words
      (the app's fixed safety reminder and referral are ignored).
    - accept_referral: questions where the guide itself defers (e.g. pesticide doses are
      "made nationally"); declining with a referral is also correct.
    """
    text = strip_boilerplate(shown_text).lower().replace("\u2019", "'")
    if not item["answerable"]:
        return {"auto_correct": not supported, "facts_hit": None}
    hits = [any(alt.lower() in text for alt in (fact if isinstance(fact, list) else [fact])) for fact in item["key_facts"]]
    correct = bool(supported and all(hits))
    if not supported and item.get("accept_referral"):
        correct = True
    return {"auto_correct": correct, "facts_hit": sum(hits), "facts_total": len(hits)}


def retrieval_fact_hit(item: dict, passages) -> bool | None:
    """Did the retrieved passages contain every key fact? (None for out-of-scope questions.)"""
    if not item.get("answerable"):
        return None
    text = " ".join(p.text for p in passages).lower().replace("\u2019", "'")
    return all(any(a.lower() in text for a in fact) for fact in item["key_facts"])


# --------------------------------------------------------------- environment
def describe_environment(fw: Fieldwise) -> dict:
    env = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "glimmer": {"base_url": settings.glimmer_base_url, "served_name": settings.glimmer_model,
                    "reasoning_strength": settings.glimmer_reasoning_strength,
                    "max_tokens": settings.glimmer_max_tokens, "seed": settings.glimmer_seed},
        "spark": {"model": settings.spark_model, "reasoning_effort": settings.spark_reasoning_effort or "server default",
                  "usd_per_m": [settings.spark_usd_per_m_input, settings.spark_usd_per_m_cached, settings.spark_usd_per_m_output]},
        "retrieval_k": settings.retrieval_k, "guide_chunks": len(fw.index),
        "fill_in_by_hand": {"device": "e.g. MacBook Pro M3 Pro, 36 GB unified memory", "gguf_file": "",
                            "llama_cpp_build": "", "network": "e.g. home fibre, Kuala Lumpur"},
    }
    try:
        root = settings.glimmer_base_url.rsplit("/v1", 1)[0]
        h = httpx.Client(trust_env=False, timeout=5)
        env["glimmer"]["models_endpoint"] = h.get(f"{settings.glimmer_base_url}/models").json()
        props = h.get(f"{root}/props")
        if props.status_code == 200:
            p = props.json()
            env["glimmer"]["server_props"] = {k: p.get(k) for k in ("build_info", "model_path", "n_ctx", "total_slots") if k in p}
    except Exception as e:  # noqa: BLE001
        env["glimmer"]["probe_error"] = str(e)
    return env


def glimmer_is_stub() -> bool:
    try:
        data = httpx.get(f"{settings.glimmer_base_url}/models", timeout=5, trust_env=False).json()
        return any("FAKE" in m.get("id", "") for m in data.get("data", []))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------- runner
def done_ids(path: Path) -> set[str]:
    """Items with a finished record. Items whose last record is an error (timeout, outage, missing key)
    are NOT done, so re-running the same command retries them; analyze.py keeps the last record per id."""
    if not path.exists():
        return set()
    last: dict[str, dict] = {}
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["id"]] = r
    return {i for i, r in last.items() if not r.get("error")}


def run_item(fw: Fieldwise, task: str, setup: str, item: dict) -> dict:
    rec: dict = {"id": item["id"], "split": item["split"], "setup": setup, "task": task,
                 "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}   # when the request was sent (UTC)
    t0 = time.perf_counter()
    try:
        if task == "images":
            jpeg = prepare(Path(item["path"]).read_bytes()).jpeg
            rec["gold"] = item["label"]
            if setup == "local":
                r = fw.local_dx(jpeg, item["crop"])
                rec.update(pred=r["label"], confidence=r["confidence"], probabilities=r["probabilities"],
                           confidence_source=r["confidence_source"], latency_s=r["latency_s"],
                           finish_reason=r["finish_reason"], usage=r["usage"], raw=r["raw"][-400:], cost_usd=0.0)
            else:
                try:
                    r = fw.cloud_dx(jpeg, item["crop"])
                    rec.update(pred=r["label"], latency_s=r["latency_s"], cost_usd=r["cost_usd"], cached=r["cached"],
                               usage=r["usage"], raw=r["raw"][-400:])
                except SparkBlocked as e:
                    # A content-filter refusal is a real outcome of the cloud setup (the app shows "Unsure"),
                    # not an outage, so it is scored and not retried.
                    rec.update(pred=None, blocked=True, blocked_reason=str(e)[:400], cost_usd=0.0,
                               latency_s=time.perf_counter() - t0)
            rec["correct"] = rec["pred"] == item["label"]
            rec["abstained"] = rec["pred"] is None
        else:
            crop = item.get("crop")
            res = fw.index.search(f"{crop or ''} {item['question']}", k=settings.retrieval_k)
            rec["retrieval_fact_hit"] = retrieval_fact_hit(item, res.passages)
            if setup == "local":
                r = fw.local_qa(item["question"], crop)
                shown = r["report"].text if r["report"] else ""
                rec.update(confidence=r["confidence"], p_supported_choice=r["p_supported_choice"],
                           coverage=r["coverage"], confidence_source=r["confidence_source"],
                           latency_s=r["latency_s"], finish_reason=r["finish_reason"], usage=r["usage"], cost_usd=0.0)
            else:
                try:
                    r = fw.cloud_qa(item["question"], res.passages, res.coverage, crop)
                    shown = r["report"].text if r["report"] else ""
                    rec.update(latency_s=r["latency_s"], cost_usd=r["cost_usd"], cached=r["cached"], usage=r["usage"])
                except SparkBlocked as e:
                    # Scored as the app behaves: no answer, the farmer gets the extension-officer referral.
                    r = {"supported": False, "report": None}
                    shown = REFERRAL
                    rec.update(blocked=True, blocked_reason=str(e)[:400], cost_usd=0.0,
                               latency_s=time.perf_counter() - t0)
            rep = r["report"]
            rec.update(supported=r["supported"], answer_shown=shown, removed_sentences=rep.removed_sentences if rep else [],
                       cited=rep.cited if rep else [], answerable=item["answerable"],
                       qtype=item.get("type"), invented_quantity=bool(rep and rep.removed_sentences))
            g = grade_qa(item, r["supported"], shown)
            rec.update(g, correct=g["auto_correct"], graded_by="auto")
    except SparkAccountBlocked:
        raise                                   # stop the whole run; see main()
    except (GlimmerUnavailable, SparkUnavailable, OSError) as e:
        rec.update(error=str(e), correct=False, latency_s=time.perf_counter() - t0)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["images", "qa"], required=True)
    ap.add_argument("--setup", choices=["local", "cloud", "both"], required=True)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--manifest", type=Path, default=HERE / "data" / "images_manifest.csv")
    ap.add_argument("--image-root", type=Path, default=settings.datasets_dir / "CCMT")
    ap.add_argument("--questions", type=Path, default=HERE / "data" / "questions.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for result files, e.g. 'smoke'")
    ap.add_argument("--allow-stub", action="store_true", help="smoke-test against scripts/fake_glimmer.py")
    ap.add_argument("--no-strict", action="store_true")
    ap.add_argument("--max-refusals", type=int, default=3,
                    help="stop a cloud run after this many content-filter refusals; repeated refusals can get a "
                         "Meta API account restricted")
    args = ap.parse_args()
    object.__setattr__(settings, "spark_max_refusals", 0)   # the runner has its own limit (--max-refusals)
    if settings.glimmer_replay:
        sys.exit("FIELDWISE_GLIMMER_REPLAY=1 is a demo mode that replays earlier results. "
                 "Set it to 0 (in .env or the shell) before running the benchmark.")

    setups = ["local", "cloud"] if args.setup == "both" else [args.setup]
    if "local" in setups and glimmer_is_stub() and not args.allow_stub:
        sys.exit("Glimmer endpoint is the FAKE stub. Start the real model (or pass --allow-stub for a smoke test).")
    tag = args.tag or ("stub" if glimmer_is_stub() else "")

    fw = Fieldwise()
    items = load_images(args.manifest, args.split, args.image_root) if args.task == "images" else load_questions(args.questions, args.split, not args.no_strict)
    if args.limit:
        items = items[: args.limit]

    RESULTS.mkdir(exist_ok=True)
    # One environment file per task and setup, so a Spark run on the laptop and a Glimmer run on another
    # machine each keep their own device/model record.
    env_name = f"environment_{args.task}_{args.setup}{('_' + tag) if tag else ''}.json"
    (RESULTS / env_name).write_text(json.dumps(describe_environment(fw), indent=2))

    for setup in setups:
        out = RESULTS / f"{args.task}_{setup}{('_' + tag) if tag else ''}.jsonl"
        skip = done_ids(out)
        todo = [it for it in items if it["id"] not in skip]
        print(f"[{setup}] {len(todo)} to run, {len(skip)} already done -> {out.name}")
        with out.open("a", encoding="utf-8") as fh:
            refusals = 0
            for n, item in enumerate(todo, start=1):
                try:
                    rec = run_item(fw, args.task, setup, item)
                except SparkAccountBlocked as e:
                    sys.exit(f"\nStopped: {e}\nThe Meta API has restricted this key. Don't retry; contact Meta / the "
                             "organisers. Finished items are saved and the run resumes once access is restored.")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                refusals += bool(rec.get("blocked"))
                if setup == "cloud" and refusals >= args.max_refusals:
                    print(f"\nStopped after {refusals} content-filter refusals in this run, to protect the Meta API "
                          "account (repeated refusals can get it restricted). Re-run with a higher --max-refusals "
                          "only if you're sure.")
                    break
                status = "ERR " + rec["error"][:60] if rec.get("error") else ("ok " if rec["correct"] else "x  ")
                print(f"  {n:4d}/{len(todo)} {status} {item['id']}  {rec.get('latency_s', 0):6.1f}s"
                      f"  ${rec.get('cost_usd') or 0:.4f}")
        recs = {}
        for line in out.read_text("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["id"]] = r
        errs = [i for i, r in recs.items() if r.get("error")]
        spend = sum(r.get("cost_usd") or 0 for r in recs.values())
        print(f"[{setup}] {len(recs) - len(errs)} done, {len(errs)} errors, API spend so far ${spend:.2f}"
              + ("  -> re-run the same command to retry the errors" if errs else ""))


if __name__ == "__main__":
    main()
