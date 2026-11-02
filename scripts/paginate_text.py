"""Recover printed page numbers from a guide's extracted text.

Some PDF text extractions lose page breaks but keep the printed page numbers
as header/footer lines ("7", "6 Cassava farmer field schools: ...",
"Part 1: 1. Importance of cassava 7"). This finds those lines, keeps only
numbers that continue the sequence (so a "5" inside a table is ignored), and
writes the text with '=== page N ===' markers that ingest_guides.py reads.

    python scripts/paginate_text.py guide.txt out.txt --header "Cassava farmer field schools"

Citations then use the printed page numbers a farmer or judge can look up.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def find_markers(lines: list[str], header: str | None, max_jump_bare: int = 4, max_jump_header: int = 8) -> dict[int, int]:
    """Map line index -> printed page number for accepted page markers.

    A bare number line ("19") is weak evidence, so it may only advance the
    sequence by a few pages. A line carrying the running header is strong
    evidence and may skip unnumbered divider or full-page-figure pages.
    """
    bare = re.compile(r"^\s*(\d{1,3})\s*$")
    strong = [re.compile(r"^\s*(?:Part|Annex|Chapter)\s+\w+[:.].*?\s(\d{1,3})\s*$")]
    if header:
        h = re.escape(header)
        strong.append(re.compile(rf"^\s*(\d{{1,3}})\s+{h}", re.IGNORECASE))
        strong.append(re.compile(rf"^\s*{h}.*?\s(\d{{1,3}})\s*$", re.IGNORECASE))

    markers: dict[int, int] = {}
    prev = 0
    for i, line in enumerate(lines):
        m = bare.match(line)
        limit = max_jump_bare
        if not m:
            m = next((mm for p in strong if (mm := p.match(line))), None)
            limit = max_jump_header
        if not m:
            continue
        n = int(m.group(1))
        if prev < n <= prev + limit:
            markers[i] = n
            prev = n
    return markers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--header", default=None, help="running-header text that follows/precedes page numbers")
    args = ap.parse_args()

    lines = args.src.read_text("utf-8", errors="replace").splitlines()
    markers = find_markers(lines, args.header)
    out = []
    for i, line in enumerate(lines):
        if i in markers:
            out.append(f"=== page {markers[i]} ===")
            continue  # drop the header/footer line itself
        out.append(line)
    args.out.write_text("\n".join(out), "utf-8")

    pages = sorted(markers.values())
    missing = sorted(set(range(pages[0], pages[-1] + 1)) - set(pages)) if pages else []
    print(f"{len(pages)} page markers, pages {pages[0] if pages else '-'}..{pages[-1] if pages else '-'}; "
          f"missing: {missing[:40]}{' ...' if len(missing) > 40 else ''}")


if __name__ == "__main__":
    main()
