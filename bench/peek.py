"""Look inside a benchmark results file: what each model answered and why it was marked wrong.

    python bench/peek.py images_cloud_smoke            # all records in bench/results/images_cloud_smoke.jsonl
    python bench/peek.py qa_cloud --wrong --n 20       # only the first 20 wrong answers

Shows the gold label, the prediction, the cost and the last part of the raw model output,
so you can tell a genuinely wrong diagnosis from a formatting problem (prediction "None").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fieldwise.safety import PPE_REMINDER  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="results file name without .jsonl, e.g. images_cloud_smoke")
    ap.add_argument("--wrong", action="store_true", help="only show items marked wrong")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--chars", type=int, default=240, help="how much of the raw output to show")
    args = ap.parse_args()

    path = RESULTS / f"{args.name.removesuffix('.jsonl')}.jsonl"
    if not path.exists():
        raise SystemExit(f"No file {path}. Files there: {', '.join(p.stem for p in RESULTS.glob('*.jsonl'))}")
    recs: dict[str, dict] = {}
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["id"]] = r                      # last record per id, as analyze.py does
    rows = [r for r in recs.values() if not (args.wrong and r.get("correct"))][: args.n]

    unparsed = 0
    for r in rows:
        mark = "ERR" if r.get("error") else ("ok " if r.get("correct") else "x  ")
        if r.get("task") == "images" or "gold" in r:
            what = f"gold {r.get('gold')}  ->  pred {r.get('pred')}"
            unparsed += r.get("pred") is None and not r.get("error")
        else:
            what = (f"answerable={r.get('answerable')} supported={r.get('supported')} "
                    f"facts {r.get('facts_hit')}/{r.get('facts_total')} retrieval_hit={r.get('retrieval_fact_hit')}")
        conf = f"  conf {r['confidence']:.2f}" if isinstance(r.get("confidence"), (int, float)) else ""
        if r.get("confidence_source"):
            conf += f" ({r['confidence_source']})"        # should say "logprobs" for real Glimmer runs
        print(f"{mark} {r['id']}  {what}{conf}  ${r.get('cost_usd') or 0:.4f}  {r.get('latency_s', 0):.1f}s")
        if r.get("error"):
            print(f"      error: {r['error'][:args.chars]}")
        if r.get("blocked"):
            print(f"      blocked by Spark's content filter: {r.get('blocked_reason', '')[:args.chars]}")
        tail = (r.get("raw") or r.get("answer_shown") or "")
        if PPE_REMINDER in tail:                      # show the model's words, not the fixed reminder
            tail = tail.replace(PPE_REMINDER, "").strip() + "  [+ safety reminder]"
        tail = tail.replace("\n", " ").strip()
        if tail:
            print(f"      ...{tail[-args.chars:]}")
    n = len(recs)
    ok = sum(1 for r in recs.values() if r.get("correct"))
    spend = sum(r.get("cost_usd") or 0 for r in recs.values())
    sources = {r.get("confidence_source") for r in recs.values() if r.get("confidence_source")}
    if sources and sources != {"logprobs"}:
        print("\nNote: some confidences didn't come from logprobs "
              f"({', '.join(sorted(sources))}). Check llama.cpp returns logprobs before trusting the routing threshold.")
    print(f"\n{ok}/{n} correct, API spend ${spend:.4f}"
          + (f", {unparsed} with no parsed prediction (check the raw output format)" if unparsed else ""))


if __name__ == "__main__":
    main()
