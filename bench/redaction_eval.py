"""Measure on-device redaction on labelled messages and list every miss.

    python bench/redaction_eval.py
    python bench/redaction_eval.py --cases bench/data/redaction_heldout.txt --out bench/results/redaction_heldout.md

Cases come as JSON lines: {"id", "set", "text", "pii": [{"kind", "text"}], "keep": ["..."]}
or, easier to write by hand, a .txt file with one message per line and inline marks:
    [NAME: yaw] here, call [PHONE: 0244 987 654]. My {cassava} in [PLACE: Ejura] is yellow.
[KIND: text] is a personal detail that must be removed; {word} must survive. Lines starting
with # are ignored. Kinds: NAME, PHONE, EMAIL, ID, GPS, PLACE (or LOCATION), ADDRESS.
A PII item counts as caught if its text no longer appears in the output.
A "keep" string (crop names, doses) must survive, or it's a false positive.

The default cases were written by the same person who wrote the rules, so
they flatter the redactor. For an honest number, have someone else write a
held-out file without looking at redact.py, and report that one.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fieldwise.redact import redact_text  # noqa: E402

HERE = Path(__file__).resolve().parent
KINDS = {"NAME", "PHONE", "EMAIL", "ID", "GPS", "LOCATION", "ADDRESS"}
_MARK = re.compile(r"\[\s*([A-Za-z]+)\s*:\s*([^\]]+?)\s*\]|\{\s*([^}]+?)\s*\}")


def parse_marked(line: str, cid: str) -> dict:
    """'[NAME: yaw] here, my {cassava} ...' -> {"id", "text", "pii", "keep"}."""
    pii, keep = [], []

    def repl(m: re.Match) -> str:
        if m.group(3) is not None:
            keep.append(m.group(3))
            return m.group(3)
        kind = m.group(1).upper()
        kind = "LOCATION" if kind == "PLACE" else kind
        if kind not in KINDS:
            raise ValueError(f"unknown kind [{m.group(1)}: ...]; use one of NAME, PHONE, EMAIL, ID, GPS, PLACE, ADDRESS")
        pii.append({"kind": kind, "text": m.group(2)})
        return m.group(2)

    return {"id": cid, "text": _MARK.sub(repl, line), "pii": pii, "keep": keep}


def load_cases(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"No file {path}. Create it first, e.g.:  notepad {path}\n"
                 "One message per line; mark details like [NAME: yaw] and words to keep like {cassava}.")
    text = path.read_text("utf-8-sig")
    cases = []
    if path.suffix.lower() != ".txt" and text.lstrip().startswith("["):      # a JSON array is fine too
        try:
            cases = json.loads(text)
        except json.JSONDecodeError as e:
            sys.exit(f"{path.name}: {e}")
    else:
        for n, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                if path.suffix.lower() == ".txt":
                    if line.startswith("{") and '"text"' in line:
                        raise ValueError("this line is JSON, but the file is .txt. Save JSON lines as a .jsonl file, "
                                         "or write plain messages with [NAME: ...] / {word} marks")
                    cases.append(parse_marked(line, f"h{len(cases) + 1:02d}"))
                else:
                    cases.append(json.loads(line.rstrip(",")))   # tolerate a comma at the end of a line
            except (ValueError, json.JSONDecodeError) as e:
                sys.exit(f"{path.name} line {n}: {e}\n  {line}")
    if not cases:
        sys.exit(f"{path.name} has no messages yet: add one per line below the # comments.")
    check_cases(cases, path.name)
    return cases


def check_cases(cases: list[dict], name: str) -> None:
    """Stop on mistakes in the test file itself, so they aren't scored as redactor misses or over-redaction."""
    problems, seen = [], set()
    for c in cases:
        cid, text = c.get("id", "?"), c.get("text", "")
        if cid in seen:
            problems.append(f"{cid}: id used twice")
        seen.add(cid)
        for item in c.get("pii", []):
            if item.get("kind") not in KINDS:
                problems.append(f"{cid}: unknown kind {item.get('kind')!r}")
            if item.get("text", "") not in text:
                problems.append(f"{cid}: personal detail {item.get('text')!r} isn't in the message as written")
        for keep in c.get("keep", []):
            if keep not in text:
                problems.append(f"{cid}: keep word {keep!r} isn't in the message as written")
    if problems:
        sys.exit(f"Fix these in {name} first (they're test-file mistakes, not redactor results):\n  "
                 + "\n  ".join(problems))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=HERE / "data" / "redaction_cases.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "redaction_report.md")
    args = ap.parse_args()
    cases = load_cases(args.cases)

    caught, total = defaultdict(int), defaultdict(int)
    misses, false_pos = [], []
    for c in cases:
        out, _ = redact_text(c["text"])
        for item in c.get("pii", []):
            total[item["kind"]] += 1
            if item["text"] in out:
                misses.append((c["id"], item["kind"], item["text"], out))
            else:
                caught[item["kind"]] += 1
        for keep in c.get("keep", []):
            if keep not in out:
                false_pos.append((c["id"], keep, out))

    lines = [f"# Redaction evaluation ({args.cases.name}, {len(cases)} messages)", "",
             "| Kind | Caught | Total | Recall |", "|---|---|---|---|"]
    for kind in sorted(total):
        lines.append(f"| {kind} | {caught[kind]} | {total[kind]} | {caught[kind] / total[kind]:.0%} |")
    all_c, all_t = sum(caught.values()), sum(total.values())
    lines += [f"| **All** | {all_c} | {all_t} | {all_c / all_t:.0%} |" if all_t else "| **All** | 0 | 0 | n/a |", "",
              f"## Misses ({len(misses)})", ""]
    lines += [f"- `{cid}` {kind}: **{text}** survived in: _{out}_" for cid, kind, text, out in misses] or ["None."]
    lines += ["", f"## Over-redaction ({len(false_pos)})", ""]
    lines += [f"- `{cid}` removed **{keep}**: _{out}_" for cid, keep, out in false_pos] or ["None."]
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
