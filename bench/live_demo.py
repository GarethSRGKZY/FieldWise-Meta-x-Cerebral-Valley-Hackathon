"""Live proof clip: run demo questions and photos through the real app pipeline with Muse Glimmer running,
and print what the app would show, with real elapsed times. For a GPU machine (Kaggle, a rented GPU);
on Kaggle use `python kaggle/kaggle_glimmer.py demo`.

    python bench/live_demo.py                                   # a default pick
    python bench/live_demo.py q-m41 q-c22 img-maize-leaf_spot-003 --pii q-c11
        # IDs as printed in [brackets] by bench/demo_examples.py on the laptop, so the live clip
        # matches the replayed scenes. --pii puts a made-up name and phone number in front of
        # those questions to show the redaction.

The Meta API is never called: the hub is forced offline, so a low-confidence item is queued with
personal details removed, exactly as on a hub with no signal. Refuses replay mode and the stub
server, because the point is the live model. Writes bench/results/live_demo.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from fieldwise import connectivity  # noqa: E402
from fieldwise.config import settings  # noqa: E402
from fieldwise.labels import LABELS  # noqa: E402
from fieldwise.pipeline import Fieldwise  # noqa: E402
from fieldwise.queue import EscalationQueue  # noqa: E402

PII = "My name is Kofi Mensah, call me on 0244 123 456. "
OUT = HERE / "results" / "live_demo.jsonl"
RULE = "-" * 78
KINDS = {"NAME": "name", "PHONE": "phone number", "EMAIL": "email", "ID": "ID number", "GPS": "GPS position",
         "LOCATION": "place name", "ADDRESS": "address", "PHOTO_GPS": "photo GPS tag"}


def fmt_conf(c: float | None) -> str:
    """Rounded down, so 0.8999 shows as 0.89 next to a 0.90 threshold, matching the routing."""
    return "none" if c is None else f"{math.floor(c * 100) / 100:.2f}"


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return "no NVIDIA GPU found"
    return f"{out[0]} x{len(out)}" if out else "no NVIDIA GPU found"


def load_results(name: str) -> dict[str, dict]:
    out = {}
    path = HERE / "results" / f"{name}.jsonl"
    if path.exists():
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    out[r["id"]] = r
    return out


def default_pick(questions: dict, photos: dict) -> tuple[list[str], list[str], list[str]]:
    """Without IDs: from earlier Glimmer results if present (2 confident, 1 low-confidence with
    the name/number, 1 out of scope, 2 photos), else the shortest questions of each kind."""
    qa, dx = load_results("qa_local"), load_results("images_local")
    with_crop = [q for q in questions.values() if q.get("crop")]
    by_len = lambda q: len(q["question"])  # noqa: E731
    if qa:
        conf = lambda q: qa.get(q["id"], {}).get("confidence") or 0  # noqa: E731
        sure = sorted([q for q in with_crop if q["answerable"] and conf(q) >= settings.qa_threshold
                       and qa.get(q["id"], {}).get("correct")], key=by_len)[:2]
        unsure = sorted([q for q in with_crop if q["answerable"] and q["id"] in qa and conf(q) < settings.qa_threshold], key=by_len)[:1]
    else:
        sure = [min((q for q in with_crop if q["answerable"] and q["crop"] == c), key=by_len, default=None) for c in ("maize", "cassava")]
        sure, unsure = [q for q in sure if q], []
    oos = sorted([q for q in with_crop if not q["answerable"]], key=by_len)[:1]
    pics = [p for p in photos.values() if not p["label"].endswith("/healthy") and p["split"] == "test"]
    if dx:
        pics = sorted((p for p in pics if p["id"] in dx), key=lambda p: -(dx[p["id"]].get("confidence") or 0))
        pics = pics[:1] + pics[-1:]          # most and least confident
    return [q["id"] for q in sure + unsure + oos], [p["id"] for p in pics[:2]], [q["id"] for q in unsure]


def show_ask(fw: Fieldwise, q: dict, text: str) -> dict:
    t0 = time.perf_counter()
    r = fw.ask(text, q["crop"])
    wall = time.perf_counter() - t0
    conf = fmt_conf(r.confidence)
    glimmer_s = (r.local or {}).get("latency_s")
    print(f"Muse Glimmer: {f'{glimmer_s:.1f} s' if glimmer_s else 'no answer'}  ·  confidence {conf} "
          f"(stays on the hub at {settings.qa_threshold:.2f} or more)  ·  route: {r.route}")
    print(f"Answered by: {r.answered_by}")
    print("Answer shown to the farmer:")
    for line in r.answer.strip().splitlines():
        print("   " + line)
    for s in r.sources:
        print(f"   [S{s['n']}] {s['title']}" + (f", page {s['page']}" if s.get("page") else ""))
    for f in r.safety_flags:
        print(f"   Safety check: {f}")
    if r.redactions:
        print("Removed before anything could leave the hub: " + ", ".join(f"{n} {KINDS.get(k, k.lower())}{'s' if n > 1 else ''}" for k, n in r.redactions.items()))
    if r.queue_id:
        item = next((i for i in fw.queue.pending(limit=500) if i["id"] == r.queue_id), None)
        if item:
            print(f"Queued for Muse Spark (sent when signal returns): \"{item['payload']['question']}\"")
    if r.note:
        print(f"Note: {r.note}")
    return {"kind": "ask", "id": q["id"], "typed": text, "wall_s": round(wall, 2), **asdict(r)}


def show_photo(fw: Fieldwise, p: dict, root: Path) -> dict:
    data = (root / p["path"]).read_bytes()
    t0 = time.perf_counter()
    r = fw.diagnose(data, p["crop"])
    wall = time.perf_counter() - t0
    conf = fmt_conf(r.confidence)
    glimmer_s = (r.local or {}).get("latency_s")
    gold = LABELS[p["label"]].name if p["label"] in LABELS else p["label"]
    print(f"Muse Glimmer: {f'{glimmer_s:.1f} s' if glimmer_s else 'no answer'}  ·  confidence {conf} "
          f"(stays on the hub at {settings.dx_threshold:.2f} or more)  ·  route: {r.route}")
    print(f"Diagnosis shown: {r.label_name}   (dataset label: {gold})")
    top = sorted(r.probabilities.items(), key=lambda kv: -kv[1])[:3]
    if top:
        print("Considered: " + ", ".join(f"{k.split('/')[-1].replace('_', ' ')} {v:.0%}" for k, v in top))
    removed = [k for k in ("exif_removed", "gps_removed") if r.photo.get(k)]
    if removed:
        print("Photo metadata removed on the hub: " + ", ".join(k.split("_")[0].upper() for k in removed))
    if r.queue_id:
        print("Queued for Muse Spark (EXIF-free copy, sent when signal returns).")
    if r.note:
        print(f"Note: {r.note}")
    rec = asdict(r)
    rec.pop("local", None)
    return {"kind": "photo", "id": p["id"], "gold": p["label"], "wall_s": round(wall, 2), **rec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="question IDs (q-...) and photo IDs from the manifest")
    ap.add_argument("--pii", nargs="*", default=None, help="question IDs to prefix with a made-up name and phone number")
    ap.add_argument("--allow-stub", action="store_true", help="only for testing this script")
    args = ap.parse_args()

    if settings.glimmer_replay:
        sys.exit("FIELDWISE_GLIMMER_REPLAY=1 is on. This clip must use the live model: set it to 0.")
    connectivity.set_forced_offline(True)                  # never call the Meta API from here
    from run_bench import glimmer_is_stub
    fw = Fieldwise(queue=EscalationQueue(settings.data_dir / "live_demo_queue.sqlite3"))
    ok, detail = fw.glimmer.health_detail()
    if not ok:
        sys.exit(f"Muse Glimmer isn't ready: {detail}")
    if glimmer_is_stub() and not args.allow_stub:
        sys.exit("That's the FAKE stub server, not Glimmer. Start the real model.")

    questions = {q["id"]: q for q in (json.loads(l) for l in (HERE / "data" / "questions.jsonl").read_text("utf-8").splitlines() if l.strip())}
    man = HERE / "data" / "images_manifest.csv"
    photos = {r["id"]: r for r in csv.DictReader(man.open(newline="", encoding="utf-8"))} if man.exists() else {}
    if args.ids:
        unknown = [i for i in args.ids if i not in questions and i not in photos]
        if unknown:
            sys.exit(f"Unknown IDs: {', '.join(unknown)}. Use the IDs in [brackets] from bench/demo_examples.py.")
        q_ids = [i for i in args.ids if i in questions]
        p_ids = [i for i in args.ids if i in photos]
        pii = set(args.pii or [])
    else:
        q_ids, p_ids, unsure = default_pick(questions, photos)
        pii = set(args.pii) if args.pii is not None else set(unsure)
    for i in q_ids:
        if not questions[i].get("crop"):
            sys.exit(f"{i} has no crop, and the app always sends one. Pick another question.")
    root = settings.datasets_dir / "CCMT"

    started = datetime.now(timezone.utc)
    print(RULE)
    print(f"LIVE RUN  {started:%Y-%m-%d %H:%M:%S} UTC")
    print(f"Muse Glimmer ({settings.glimmer_model}) served by llama.cpp on {gpu_name()}.")
    print("Hub forced offline: no Meta API calls. Low-confidence items are queued, redacted.")
    print(RULE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with OUT.open("a", encoding="utf-8") as fh:
        for kind, ids in (("ask", q_ids), ("photo", p_ids)):
            for i in ids:
                now = datetime.now(timezone.utc)
                if kind == "ask":
                    q = questions[i]
                    text = (PII if i in pii else "") + q["question"]
                    print(f"\n[{now:%H:%M:%S} UTC]  Question {i} ({q['crop']}): {text}")
                    rec = show_ask(fw, q, text)
                else:
                    p = photos[i]
                    print(f"\n[{now:%H:%M:%S} UTC]  Photo {i} ({p['crop']}): {Path(p['path']).name}")
                    rec = show_photo(fw, p, root)
                rec["ts"] = now.isoformat(timespec="seconds")
                rows.append(rec)
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()

    print("\n" + RULE)
    print(f"{'item':<34}{'route':<16}{'confidence':>11}{'seconds':>10}")
    for r in rows:
        c = fmt_conf(r["confidence"])
        print(f"{r['id']:<34}{r['route']:<16}{c:>11}{r['wall_s']:>10.1f}")
    print(f"Finished {datetime.now(timezone.utc):%H:%M:%S} UTC. Saved to {OUT.relative_to(HERE.parent)}.")


if __name__ == "__main__":
    main()
