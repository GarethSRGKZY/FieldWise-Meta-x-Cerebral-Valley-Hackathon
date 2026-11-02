"""How often does retrieval put the answer in front of the model?

    python bench/retrieval_eval.py            # dev split (use this while tuning retrieval)
    python bench/retrieval_eval.py --split test   # report once, after tuning is frozen

Two measures per answerable question, for several k:
- page hit: the cited guide page is among the top-k passages
- fact hit: every key fact (one of its alternatives) appears somewhere in the top-k passages.
  Fairer, since many facts appear on more than one page.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fieldwise.config import settings  # noqa: E402
from fieldwise.retrieval import GuideIndex  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=HERE / "data" / "questions.jsonl")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5, 8])
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    idx = GuideIndex.load(settings.index_path)
    if not len(idx):
        sys.exit(f"No guide index at {settings.index_path}. Run scripts/ingest_guides.py first.")
    rows = [json.loads(l) for l in args.questions.read_text("utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["answerable"] and (args.split == "all" or r["split"] == args.split)]

    report = {"split": args.split, "n": len(rows), "by_k": {}}
    for k in args.ks:
        page_hits = fact_hits = 0
        misses = []
        for q in rows:
            res = idx.search(f"{q.get('crop') or ''} {q['question']}", k=k)
            want = {q["source"]["page"], *q["source"].get("also_pages", [])}
            page_ok = any(p.doc_id == q["source"]["doc_id"] and p.page in want for p in res.passages)
            text = " ".join(p.text for p in res.passages).lower().replace("’", "'")
            fact_ok = all(any(a.lower() in text for a in fact) for fact in q["key_facts"])
            page_hits += page_ok
            fact_hits += fact_ok
            if not fact_ok:
                misses.append(q["id"])
        report["by_k"][k] = {"page_hit": page_hits / len(rows), "fact_hit": fact_hits / len(rows), "fact_misses": misses}
        print(f"k={k}: page hit {page_hits}/{len(rows)} ({page_hits / len(rows):.0%}), "
              f"fact hit {fact_hits}/{len(rows)} ({fact_hits / len(rows):.0%})")
        if args.show_misses:
            print("   fact misses:", " ".join(misses))
    out = HERE / "results" / f"retrieval_{args.split}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
