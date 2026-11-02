"""Pack everything the Kaggle Glimmer run needs into one zip, to upload as a PRIVATE Kaggle dataset.

    python bench/pack_for_kaggle.py            # -> fieldwise_kaggle.zip in the project folder

Contents: the code, your reviewed questions, the frozen photo manifest, the FAO guide text, and
only the 204 benchmark photos (not all of CCMT), each checked against the manifest's SHA-1.
Never included: .env (your API key), models, llama.cpp, results, caches.

Keep the Kaggle dataset private: the FAO guides allow non-commercial research use, not
republishing. CCMT photos are CC BY 4.0 (credit Mensah et al. 2023).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fieldwise.config import settings  # noqa: E402

CODE_DIRS = ["fieldwise", "app", "bench", "scripts", "kaggle"]   # app/: the `ui` step runs the real app
CODE_FILES = ["requirements.txt", "README.md"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", "results"}
SKIP_NAMES = {"questions_review.csv", ".env"}   # never ship secrets: the Kaggle script sets its own settings


def code_files() -> list[Path]:
    out = [ROOT / f for f in CODE_FILES if (ROOT / f).exists()]
    for d in CODE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            rel = p.relative_to(ROOT)
            if rel.parts[:2] == ("bench", "data"):
                continue                    # the data files are added explicitly below
            if p.is_file() and not (set(rel.parts) & SKIP_PARTS) and p.name not in SKIP_NAMES:
                out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "fieldwise_kaggle.zip")
    ap.add_argument("--allow-drafts", action="store_true", help="pack even if some questions are still draft")
    args = ap.parse_args()

    # Questions must be reviewed: the Kaggle run uses the same strict check as your laptop runs.
    qfile = ROOT / "bench" / "data" / "questions.jsonl"
    rows = [json.loads(l) for l in qfile.read_text("utf-8").splitlines() if l.strip()]
    drafts = sum(r.get("review_status") == "draft" for r in rows)
    if drafts and not args.allow_drafts:
        sys.exit(f"{drafts} questions are still draft. Finish the review and run "
                 "`python bench/review_questions.py import` first.")

    manifest = ROOT / "bench" / "data" / "images_manifest.csv"
    if not manifest.exists():
        sys.exit("No bench/data/images_manifest.csv. Run python bench/prepare_images.py first.")
    image_root = settings.datasets_dir / "CCMT"
    with manifest.open() as fh:
        images = list(csv.DictReader(fh))
    bad = []
    for r in images:
        p = image_root / r["path"]
        if not p.exists() or (r.get("sha1") and hashlib.sha1(p.read_bytes()).hexdigest() != r["sha1"]):
            bad.append(r["path"])
    if bad:
        sys.exit(f"{len(bad)} manifest photos are missing or changed under {image_root}, e.g. {bad[0]}")

    guides = ROOT / "data" / "guides"
    sources = guides / "sources.json" if (guides / "sources.json").exists() else guides / "sources.example.json"
    srcs = json.loads(sources.read_text("utf-8"))
    guide_files = [guides / s["file"] for s in srcs]
    missing = [g.name for g in guide_files if not g.exists()]
    if missing:
        sys.exit(f"Guide text missing: {missing}")

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in code_files():
            z.write(p, f"fieldwise/{p.relative_to(ROOT).as_posix()}")
        z.write(qfile, "fieldwise/bench/data/questions.jsonl")
        z.write(manifest, "fieldwise/bench/data/images_manifest.csv")
        z.write(sources, "fieldwise/data/guides/sources.json")
        for g in guide_files:
            z.write(g, f"fieldwise/data/guides/{g.name}")
        for r in images:
            z.write(image_root / r["path"], f"fieldwise/datasets/CCMT/{r['path']}")
    size = args.out.stat().st_size / 1e6
    print(f"Wrote {args.out} ({size:.0f} MB): code, {len(rows)} questions ({drafts} draft), "
          f"{len(images)} photos, {len(guide_files)} guides. No .env, keys or models inside.")
    print("Upload it at kaggle.com -> Create -> New Dataset, and keep the dataset PRIVATE.")


if __name__ == "__main__":
    main()
