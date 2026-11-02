"""Re-grade saved Q&A results with the current grader, without calling any model again.

    python bench/regrade_qa.py            # every bench/results/qa_*.jsonl (backups saved as .bak first)

Use after a grading fix. It recomputes the automatic grade from each record's stored answer, for every
setup alike, and prints which items changed so the change can be reported. Manual grades are untouched.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_bench import grade_qa  # noqa: E402


def main() -> None:
    questions = {json.loads(l)["id"]: json.loads(l) for l in (HERE / "data" / "questions.jsonl").read_text("utf-8").splitlines() if l.strip()}
    files = sorted(p for p in (HERE / "results").glob("qa_*.jsonl") if "smoke" not in p.name)
    if not files:
        sys.exit("No bench/results/qa_*.jsonl files.")
    for path in files:
        lines = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
        flips = []
        for r in lines:
            q = questions.get(r["id"])
            if q is None or r.get("error") or r.get("graded_by") == "manual" or "supported" not in r:
                continue
            g = grade_qa(q, bool(r["supported"]), r.get("answer_shown") or "")
            if g["auto_correct"] != r.get("correct"):
                flips.append(f"{r['id']} {'wrong->right' if g['auto_correct'] else 'right->wrong'}")
            r.update(g, correct=g["auto_correct"], graded_by="auto")
        backup = path.with_suffix(".jsonl.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines), "utf-8")
        print(f"{path.name}: {len(flips)} grade(s) changed" + (": " + ", ".join(flips) if flips else ""))
    print("Now re-run: python bench/analyze.py --task qa")


if __name__ == "__main__":
    main()
