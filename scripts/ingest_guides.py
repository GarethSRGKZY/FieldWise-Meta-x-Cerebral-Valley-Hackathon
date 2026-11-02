"""Build the offline guide index from data/guides/sources.json.

    python scripts/ingest_guides.py

sources.json lists each guide with its licence, so every answer can show
where it came from and whether we're allowed to use it:

    [{"id": "fao-faw-ffs", "title": "...", "file": "fao_faw_ffs_2018.pdf",
      "url": "https://...", "licence": "CC BY-NC-SA 3.0 IGO", "crops": ["maize"]}]

PDFs are read page by page (page numbers become citations). .txt/.md files
can mark pages with lines like '=== page 12 ==='.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fieldwise.config import settings  # noqa: E402
from fieldwise.retrieval import chunk_pages  # noqa: E402


def read_pages(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    text = path.read_text("utf-8", errors="replace")
    parts = re.split(r"^===\s*page\s+(\d+)\s*===\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    if len(parts) == 1:
        return [(None, text)]
    pages = []
    if parts[0].strip():
        pages.append((None, parts[0]))
    for i in range(1, len(parts), 2):
        pages.append((int(parts[i]), parts[i + 1]))
    return pages


def main() -> None:
    src_file = settings.guides_dir / "sources.json"
    if not src_file.exists():
        sys.exit(f"Missing {src_file}. Copy sources.example.json and fill it in.")
    sources = json.loads(src_file.read_text("utf-8"))
    chunks = []
    for src in sources:
        path = settings.guides_dir / src["file"]
        if not path.exists():
            print(f"  ! skipping {src['id']}: {path.name} not found")
            continue
        if not src.get("licence"):
            sys.exit(f"{src['id']} has no licence recorded. Record it before indexing.")
        if path.suffix.lower() == ".pdf":
            print(f"  ! {src['id']}: reading the PDF directly cites PDF page numbers (1 = cover), not the "
                  "printed ones in bench/data/questions.jsonl. Use the paginated .txt for benchmark runs.")
        pages = read_pages(path)
        pieces = chunk_pages(pages)
        for n, (page, text) in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{src['id']}#{n}", "doc_id": src["id"], "title": src["title"],
                "page": page, "text": text, "url": src.get("url", ""), "licence": src["licence"],
                "crops": src.get("crops", []),
            })
        print(f"  {src['id']}: {len(pages)} pages -> {len(pieces)} chunks")
    settings.index_path.parent.mkdir(parents=True, exist_ok=True)
    settings.index_path.write_text(json.dumps({"chunks": chunks}, ensure_ascii=False), "utf-8")
    print(f"Wrote {len(chunks)} chunks to {settings.index_path}")


if __name__ == "__main__":
    main()
