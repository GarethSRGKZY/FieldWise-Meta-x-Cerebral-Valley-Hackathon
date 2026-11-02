"""Review the question set in a spreadsheet instead of raw JSONL.

    python bench/review_questions.py export          # -> bench/data/questions_review.csv
    # open it in Excel / LibreOffice / Google Sheets, check each row against the guide page,
    # edit question / key_facts / page if needed, set review_status to reviewed (or reject), save as CSV
    python bench/review_questions.py import          # writes the edits back into questions.jsonl

key_facts format in the CSV: facts separated by " ; ", alternatives within a fact by " | ".
Example: "fall armyworm | spodoptera ; whorl"  = must mention (fall armyworm OR spodoptera) AND whorl.
Rows marked "reject" are dropped from questions.jsonl on import (and listed, so you can say so in the write-up).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
QFILE = HERE / "data" / "questions.jsonl"
CSVFILE = HERE / "data" / "questions_review.csv"
COLS = ["id", "split", "crop", "answerable", "question", "key_facts", "doc_id", "page", "also_pages",
        "accept_referral", "review_status", "reviewer_note"]


def facts_to_cell(facts: list) -> str:
    return " ; ".join(" | ".join(f if isinstance(f, list) else [f]) for f in facts)


def cell_to_facts(cell: str) -> list[list[str]]:
    return [[a.strip() for a in fact.split("|") if a.strip()] for fact in cell.split(";") if fact.strip()]


def _unimported_edits(rows: list[dict]) -> bool:
    """True if the sheet differs from questions.jsonl (someone has been reviewing in it)."""
    by_id = {r["id"]: r for r in rows}
    try:
        with CSVFILE.open(encoding="utf-8-sig") as fh:
            sheet = list(csv.DictReader(fh))
    except (OSError, csv.Error):
        return False
    for e in sheet:
        r = by_id.get(e.get("id"))
        if r is None:
            return True
        if (e.get("review_status", "").strip().lower() != r.get("review_status", "draft")
                or e.get("question", "").strip() != r["question"].strip()
                or (r["answerable"] and cell_to_facts(e.get("key_facts", "")) != [f if isinstance(f, list) else [f] for f in r["key_facts"]])
                or e.get("reviewer_note", "").strip() != r.get("reviewer_note", "")):
            return True
    return False


def export(rows: list[dict]) -> None:
    with CSVFILE.open("w", newline="", encoding="utf-8-sig") as fh:   # BOM so Excel opens UTF-8 correctly
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            src = r.get("source") or {}
            w.writerow({"id": r["id"], "split": r["split"], "crop": r.get("crop") or "", "answerable": r["answerable"],
                        "question": r["question"], "key_facts": facts_to_cell(r["key_facts"]),
                        "doc_id": src.get("doc_id", ""), "page": src.get("page", ""),
                        "also_pages": ",".join(str(p) for p in src.get("also_pages", [])),
                        "accept_referral": bool(r.get("accept_referral")), "review_status": r.get("review_status", "draft"),
                        "reviewer_note": r.get("reviewer_note", "")})
    print(f"Wrote {len(rows)} rows to {CSVFILE}")


def do_import(rows: list[dict]) -> None:
    by_id = {r["id"]: r for r in rows}
    with CSVFILE.open(encoding="utf-8-sig") as fh:
        edits = list(csv.DictReader(fh))
    rejected, changed = [], 0
    for e in edits:
        r = by_id.get(e["id"])
        if r is None:
            raise SystemExit(f"{e['id']} is in the CSV but not in {QFILE.name}; don't add rows in the sheet.")
        status = e["review_status"].strip().lower()
        if status not in {"draft", "reviewed", "reject"}:
            raise SystemExit(f"{e['id']}: review_status must be draft, reviewed or reject (got {status!r})")
        before = json.dumps(r, sort_keys=True)
        r["question"] = e["question"].strip()
        r["key_facts"] = cell_to_facts(e["key_facts"]) if r["answerable"] else r["key_facts"]
        if r.get("source"):                      # out-of-scope rows have no source page
            r["source"]["page"] = int(e["page"])
            if e["also_pages"].strip():
                r["source"]["also_pages"] = [int(p) for p in e["also_pages"].split(",") if p.strip()]
            else:
                r["source"].pop("also_pages", None)
        r["review_status"] = status
        ar = e.get("accept_referral", "").strip().lower()
        if ar in ("true", "1", "yes", "y"):
            r["accept_referral"] = True
        elif ar in ("false", "0", "no", "n", ""):
            r.pop("accept_referral", None)
        else:
            raise SystemExit(f"{e['id']}: accept_referral must be TRUE or FALSE (got {e['accept_referral']!r})")
        if e["reviewer_note"].strip():
            r["reviewer_note"] = e["reviewer_note"].strip()
        if status == "reject":
            rejected.append(r["id"])
        changed += json.dumps(r, sort_keys=True) != before
    kept = [r for r in rows if r["id"] not in set(rejected)]
    QFILE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), "utf-8")
    left = sum(1 for r in kept if r.get("review_status") == "draft")
    print(f"Updated {changed} rows; rejected {len(rejected)} {rejected or ''}; {len(kept)} remain, {left} still draft.")
    if len(kept) < 100:
        print("Warning: fewer than 100 questions left; the runner will refuse a strict run.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "import"])
    ap.add_argument("--force", action="store_true", help="export even if the sheet has review work not yet imported")
    args = ap.parse_args()
    rows = [json.loads(l) for l in QFILE.read_text("utf-8").splitlines() if l.strip()]
    if args.action == "export":
        if CSVFILE.exists() and not args.force and _unimported_edits(rows):
            raise SystemExit(f"{CSVFILE.name} has edits that aren't in questions.jsonl yet. Exporting would overwrite "
                             "them. Run `python bench/review_questions.py import` first (or add --force to discard them).")
        export(rows)
    else:
        try:
            do_import(rows)
        except PermissionError:
            raise SystemExit(f"Can't read {CSVFILE.name}: close it in Excel first, then run the import again.")


if __name__ == "__main__":
    main()
