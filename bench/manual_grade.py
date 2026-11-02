"""Blind manual check of the automatic Q&A grades, for both models alike.

    python bench/manual_grade.py export     # -> bench/data/manual_grading.csv (model names hidden, shuffled)
    # open it, read each answer against the key facts and the guide page, put y or n in "correct", save as CSV
    python bench/manual_grade.py import     # -> bench/data/qa_grades_manual.csv, used by analyze.py

Which answers are included: every answer the model presented as supported but the auto-grader marked
wrong (possible false negatives), plus a random 10 per model the auto-grader marked right (to measure
false positives). Rows are shuffled and show neither the model nor the automatic grade, so the
grading can't favour either. Report how many grades changed in the write-up.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fieldwise.safety import PPE_REMINDER, REFERRAL  # noqa: E402

SHEET = HERE / "data" / "manual_grading.csv"
KEY = HERE / "data" / "manual_grading_key.json"        # which row is which model; don't open while grading
OUT = HERE / "data" / "qa_grades_manual.csv"


def load(path: Path) -> dict[str, dict]:
    recs = {}
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["id"]] = r
    return recs


def export(sample_right: int, seed: int) -> None:
    qs = {json.loads(l)["id"]: json.loads(l) for l in (HERE / "data" / "questions.jsonl").read_text("utf-8").splitlines() if l.strip()}
    rng = random.Random(seed)
    rows, key = [], {}
    for setup in ("local", "cloud"):
        path = HERE / "results" / f"qa_{setup}.jsonl"
        if not path.exists():
            sys.exit(f"Missing {path.name}.")
        recs = [r for r in load(path).values() if r.get("answerable") and r.get("supported") and not r.get("error")]
        wrong = [r for r in recs if not r.get("correct")]
        right = [r for r in recs if r.get("correct")]
        for r in wrong + rng.sample(right, min(sample_right, len(right))):
            q = qs[r["id"]]
            answer = (r.get("answer_shown") or "").replace(PPE_REMINDER, "").replace(REFERRAL, "").strip()
            rows.append({"question_id": r["id"], "question": q["question"],
                         "key_facts": " ; ".join(" | ".join(f if isinstance(f, list) else [f]) for f in q["key_facts"]),
                         "guide_page": f"{q['source']['doc_id']} p.{q['source']['page']}",
                         "answer": answer, "correct": ""})
            key[str(len(rows))] = (setup, "right" if r.get("correct") else "wrong")
    order = list(range(len(rows)))
    rng.shuffle(order)
    with SHEET.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["row"] + list(rows[0]))
        w.writeheader()
        new_key = {}
        for n, i in enumerate(order, start=1):
            w.writerow({"row": n, **rows[i]})
            setup, auto = key[str(i + 1)]
            new_key[str(n)] = {"setup": setup, "id": rows[i]["question_id"], "auto_grade": auto}
    KEY.write_text(json.dumps(new_key, indent=1))
    print(f"Wrote {len(rows)} rows to {SHEET.name}. Fill 'correct' with y or n using the key facts and the guide page.\n"
          "Judge meaning, not exact words. Don't open manual_grading_key.json until you've finished.")


def do_import() -> None:
    key = json.loads(KEY.read_text())
    try:
        with SHEET.open(encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    except PermissionError:
        sys.exit(f"Close {SHEET.name} in Excel first.")
    out, todo, changed = [], [], 0
    for r in rows:
        v = r["correct"].strip().lower()
        if v not in ("y", "n", "yes", "no"):
            todo.append(r["row"])
            continue
        k = key[r["row"]]
        ok = v in ("y", "yes")
        changed += ok != (k["auto_grade"] == "right")
        out.append({"setup": k["setup"], "id": k["id"], "correct": "yes" if ok else "no"})
    if todo:
        sys.exit(f"{len(todo)} rows still have no y/n in 'correct' (rows {', '.join(todo[:10])}...). Finish them first.")
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["setup", "id", "correct"])
        w.writeheader()
        w.writerows(out)
    by = {s: sum(1 for o in out if o["setup"] == s) for s in ("local", "cloud")}
    print(f"Wrote {OUT.name}: {len(out)} manual grades (Glimmer {by['local']}, Spark {by['cloud']}); "
          f"{changed} differ from the automatic grade. Now run: python bench/analyze.py --task qa")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "import"])
    ap.add_argument("--sample-right", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    export(args.sample_right, args.seed) if args.action == "export" else do_import()


if __name__ == "__main__":
    main()
