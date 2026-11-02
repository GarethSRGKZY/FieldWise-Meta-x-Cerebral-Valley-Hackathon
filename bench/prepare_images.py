"""Freeze the image benchmark: a stratified dev/test split of CCMT raw images.

    python bench/prepare_images.py                      # reads datasets/CCMT/ by default
    python bench/prepare_images.py --root "/elsewhere/CCMT"

Crops default to FIELDWISE_CROPS (maize and cassava: 12 classes, so 60 dev
and 144 test images at the default sizes).

Source: CCMT, "Dataset for Crop Pest and Disease Detection" (Mensah et al.,
Data in Brief 2023), Mendeley Data doi:10.17632/bwh3zbpkpv.1, CC BY 4.0.
Download it yourself; it is not redistributed here.

Rules this script enforces:
- Raw images only. Anything under a folder with "augment" in its name is
  skipped, because augmented copies of a test image would leak into dev.
- Exact duplicates (same bytes) are dropped.
- Fixed seed, so the split is reproducible. Run once, commit the manifest,
  and don't re-sample after you've seen results.
- Paths are stored relative to --root, with each file's SHA-1, so the same
  manifest works on another machine (e.g. a borrowed hub) and run_bench.py
  can check it is reading exactly the same images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fieldwise.config import settings  # noqa: E402
from fieldwise.labels import ALL_CROPS, CROPS, match_label  # noqa: E402

SOURCE = "CCMT (Mensah et al. 2023), CC BY 4.0, doi:10.17632/bwh3zbpkpv.1"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def classify(path: Path, root: Path) -> tuple[str, str] | None:
    parts = [p for p in path.relative_to(root).parts[:-1]]
    if any("augment" in p.lower() for p in parts):
        return None
    for i, part in enumerate(parts):
        crop = part.strip().lower()
        if crop in ALL_CROPS:
            for cls in parts[i + 1:]:
                lab = match_label(crop, cls)
                if lab:
                    return crop, lab.key
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=settings.datasets_dir / "CCMT",
                    help="folder holding the raw CCMT crop folders (default: datasets/CCMT)")
    ap.add_argument("--crops", nargs="+", default=list(CROPS))
    ap.add_argument("--dev-per-class", type=int, default=5)
    ap.add_argument("--test-per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "images_manifest.csv")
    args = ap.parse_args()
    if not args.root.is_dir():
        sys.exit(f"No folder at {args.root}. Unzip CCMT there or pass --root.")

    by_class: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    seen: set[str] = set()
    unmatched: dict[str, int] = defaultdict(int)
    for path in sorted(args.root.rglob("*")):
        if path.suffix.lower() not in EXTS:
            continue
        hit = classify(path, args.root)
        if not hit:
            unmatched[str(path.parent.relative_to(args.root))] += 1
            continue
        crop, key = hit
        if crop not in args.crops:
            continue
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        by_class[key].append((path, digest))

    if not by_class:
        sys.exit("No images matched. Check --root points at the raw CCMT folders (Crop/Class/*.jpg).")

    rng = random.Random(args.seed)
    rows = []
    for key in sorted(by_class):
        files = sorted(by_class[key])
        rng.shuffle(files)
        need = args.dev_per_class + args.test_per_class
        if len(files) < need:
            print(f"  ! {key}: only {len(files)} images (wanted {need})")
        for i, (f, digest) in enumerate(files[:need]):
            split = "dev" if i < args.dev_per_class else "test"
            rows.append({"id": f"img-{key.replace('/', '-')}-{i:03d}", "split": split, "crop": key.split("/")[0],
                         "label": key, "path": f.relative_to(args.root).as_posix(), "sha1": digest, "source": SOURCE})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_dev = sum(r["split"] == "dev" for r in rows)
    print(f"Wrote {len(rows)} images ({n_dev} dev, {len(rows) - n_dev} test, {len(by_class)} classes) to {args.out}")
    if unmatched:
        top = sorted(unmatched.items(), key=lambda kv: -kv[1])[:8]
        print("Folders not matched to a class (check these aren't classes you wanted):")
        for folder, n in top:
            print(f"  {n:5d}  {folder}")


if __name__ == "__main__":
    main()
