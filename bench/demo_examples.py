"""Pick demo examples for each scene of the video from the real benchmark results.

    python bench/demo_examples.py            # run on the laptop that holds bench/results and data/cache

For replay mode (FIELDWISE_GLIMMER_REPLAY=1) the demo can only use benchmark questions and photos, and
a cloud scene only works with the network blocked if Spark's answer is already cached for the exact
request the app will send. This script checks both, so nothing on the shortlist fails on camera.
No network calls; nothing is sent anywhere.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from fieldwise import redact  # noqa: E402
from fieldwise.cache import DiskCache, request_key  # noqa: E402
from fieldwise.config import settings  # noqa: E402
from fieldwise.imageprep import prepare  # noqa: E402
from fieldwise.prompts import DX_SYSTEM, QA_SYSTEM, dx_user_prompt, qa_user_prompt  # noqa: E402
from fieldwise.providers.spark import SparkClient  # noqa: E402
from fieldwise.retrieval import GuideIndex  # noqa: E402

RESULTS = HERE / "results"
SHOW = 5


def last_by_id(name: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    path = RESULTS / f"{name}.jsonl"
    if path.exists():
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    out[r["id"]] = r
    return out


def spark_cached(spark: SparkClient, cache: DiskCache, system: str, user: str, jpeg: bytes | None) -> bool:
    return cache.get(request_key(spark._body(system, user, jpeg))) is not None


def line(*parts) -> None:
    print("  " + "  ".join(str(p) for p in parts if p != ""))


def main() -> None:
    qa_local, qa_cloud = last_by_id("qa_local"), last_by_id("qa_cloud")
    dx_local, dx_cloud = last_by_id("images_local"), last_by_id("images_cloud")
    if not (qa_local or dx_local):
        sys.exit(f"No Glimmer results in {RESULTS} (qa_local.jsonl / images_local.jsonl). Unzip the Kaggle results there first.")
    questions = {q["id"]: q for q in (json.loads(l) for l in (HERE / "data" / "questions.jsonl").read_text("utf-8").splitlines() if l.strip())}
    index = GuideIndex.load(settings.index_path)
    spark, cache = SparkClient(), DiskCache(settings.cache_dir, "spark")
    qt, dt = settings.qa_threshold, settings.dx_threshold
    print(f"Thresholds from .env: questions {qt}, photos {dt}. Replay mode is "
          f"{'ON' if settings.glimmer_replay else 'OFF (set FIELDWISE_GLIMMER_REPLAY=1 in .env for the demo)'}.\n")

    # ------------------------------------------------------------------ questions
    local_ok, escalate, out_of_scope, low = [], [], [], []
    for qid, r in qa_local.items():
        q = questions.get(qid)
        if not q or not q.get("crop"):
            continue                     # the app always sends a crop; crop-less questions can't be replayed
        conf = r.get("confidence") or 0.0
        if not q["answerable"]:
            out_of_scope.append((q, r))
            continue
        if conf >= qt and r.get("correct") and r.get("supported") and r.get("cited"):
            local_ok.append((q, r))
        elif conf < qt:
            low.append((q, r))
            c = qa_cloud.get(qid)
            red, _ = redact.redact_text(q["question"])
            res = index.search(f"{q['crop']} {q['question']}", k=settings.retrieval_k)
            hit = spark_cached(spark, cache, QA_SYSTEM, qa_user_prompt(red, res.passages, q["crop"]), None)
            if hit and c and c.get("correct") and not c.get("blocked"):
                escalate.append((q, r, c))

    print(f"SCENE: question answered on the hub, airplane mode on ({len(local_ok)} candidates)")
    for q, r in sorted(local_ok, key=lambda t: (len(t[0]["question"]), -t[1]["confidence"]))[:SHOW]:
        line(f"[{q['id']}]", f"crop={q['crop']}", f"conf {r['confidence']:.2f}", f"Kaggle {r.get('latency_s', 0):.0f}s")
        line("   type:", q["question"])
    print(f"\nSCENE: low confidence on the hub, escalated to Spark (network on; Spark's answer is cached) ({len(escalate)})")
    for q, r, c in sorted(escalate, key=lambda t: (t[1].get("correct", False), len(t[0]["question"])))[:SHOW]:
        line(f"[{q['id']}]", f"crop={q['crop']}", f"Glimmer conf {r['confidence']:.2f}",
             f"Glimmer {'right' if r.get('correct') else 'wrong'}, Spark right")
        line("   type:", q["question"])
    if escalate:
        print("  Type it exactly (no extra words), or the request changes and the cached Spark answer won't match.")
    print("\nSCENE: no signal, so a low-confidence question is queued with personal details removed")
    if low:
        q, r = min(low, key=lambda t: len(t[0]["question"]))
        print(f"  Any low-confidence question works ({len(low)}), with a made-up name and number in front, e.g. [{q['id']}] "
              f"crop={q['crop']}, Glimmer conf {r['confidence']:.2f}:")
        print(f"   type: My name is Kofi Mensah, call me on 0244 123 456. {q['question']}")
        print("  The app shows Glimmer's tentative answer and queues a copy with the name and number removed.")
    else:
        print("  None: every recorded question is above the threshold.")
    print(f"\nSCENE: question the guides don't cover, so the farmer is referred to an extension officer ({len(out_of_scope)})")
    for q, r in sorted(out_of_scope, key=lambda t: len(t[0]["question"]))[:3]:
        what = "declined on the hub" if (r.get("confidence") or 0) >= qt else "low confidence: escalates/queues"
        line(f"[{q['id']}]", f"crop={q['crop']}", f"conf {r.get('confidence', 0):.2f}", what,
             "Glimmer declined correctly" if r.get("correct") else "Glimmer answered anyway")
        line("   type:", q["question"])

    # ------------------------------------------------------------------ photos
    man_path = HERE / "data" / "images_manifest.csv"
    manifest = ({row["id"]: row for row in csv.DictReader(man_path.open(newline="", encoding="utf-8"))}
                if man_path.exists() else {})
    if not (manifest and dx_local):
        print(f"\nPhotos: skipped ({'no images_local.jsonl results' if manifest else 'no ' + man_path.name}).")
    root = settings.datasets_dir / "CCMT"
    dx_ok, dx_up = [], []
    for pid, r in dx_local.items():
        m = manifest.get(pid)
        if not m:
            continue
        conf = r.get("confidence") or 0.0
        if conf >= dt and r.get("correct") and not str(r.get("gold", "")).endswith("/healthy"):
            dx_ok.append((m, r))
        elif conf < dt:
            c = dx_cloud.get(pid)
            path = root / m["path"]
            if c and c.get("correct") and not c.get("blocked") and path.exists():
                jpeg = prepare(path.read_bytes()).jpeg
                if spark_cached(spark, cache, DX_SYSTEM, dx_user_prompt(m["crop"], None), jpeg):
                    dx_up.append((m, r, c))
    print(f"\nSCENE: leaf photo diagnosed on the hub ({len(dx_ok)} candidates)")
    for m, r in sorted(dx_ok, key=lambda t: -t[1]["confidence"])[:SHOW]:
        line(f"[{m['id']}]", f"crop={m['crop']}", m["label"], f"conf {r['confidence']:.2f}")
        line("   file:", root / m["path"])
    print(f"\nSCENE: photo escalated to Spark (network on; Spark's answer is cached) ({len(dx_up)})")
    for m, r, c in sorted(dx_up, key=lambda t: t[1].get("correct", False))[:SHOW]:
        line(f"[{m['id']}]", f"crop={m['crop']}", m["label"], f"Glimmer conf {r['confidence']:.2f}",
             f"Glimmer said {r.get('pred')}, Spark right")
        line("   file:", root / m["path"])
    if dx_up:
        print("  Upload the original file (not a copy a phone or chat app re-saved), leave the note empty,\n"
              "  or the request changes and the cached Spark answer won't match.")

    sam = settings.cache_dir / "sam"
    n_sam = len(list(sam.glob("*.json"))) if sam.exists() else 0
    print(f"\nSAM 3.1: {n_sam} cached responses. Each photo needs 2 (leaf + lesion), so the severity overlay "
          + ("may work for photos measured before the block." if n_sam else "can't be shown until Meta restores access."))
    print("\nAvoid the 'What should I do about it?' button in replay mode: the question it writes isn't a benchmark question.")


if __name__ == "__main__":
    main()
