"""Choose SAM 3.1's prompt phrases by measurement, without touching the test photos.

SAM 3.1 segments whatever a short noun phrase names, and our first phrases were guesses: on 30 Oct
"leaf spot" found nothing on a corn gray-leaf-spot photo, and "leaf" found nothing on a CCMT maize
photo. This script tries candidate phrases on photos that are NOT used for reporting:

  damage phrases: PlantSeg val (or train) photos, scored by IoU against PlantSeg's lesion masks
  leaf phrases:   CCMT dev-split photos (no leaf masks exist, so: how often a leaf is found and how
                  much of the photo it covers), with overlays saved for you to look at

Then report the chosen damage phrase ONCE on the PlantSeg test photos with bench/sam_eval.py.

    python bench/sam_phrases.py                       # shows the plan and its cost; no calls
    python bench/sam_phrases.py --yes                 # runs it (default ~$0.24)
    python bench/sam_phrases.py --yes --lesion-phrases spot lesion "gray spot" --n-lesion 10

Replies are cached, so re-running with more phrases only pays for the new ones.
PlantSeg is CC BY-NC-ND: numbers only, no PlantSeg overlays are saved. CCMT is CC BY 4.0.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from fieldwise.config import settings  # noqa: E402
from fieldwise.imageprep import prepare  # noqa: E402
from fieldwise.providers.sam import SamClient, SamUnavailable, overlay_png, rasterize  # noqa: E402
from sam_eval import find_meta, gt_mask, iou, read_meta, stop_on  # noqa: E402

LESION_PHRASES = ["leaf spot", "spot", "lesion", "brown spot", "gray spot", "disease spot", "dead tissue", "brown patch"]
LEAF_PHRASES = ["leaf", "leaves", "{crop} leaf", "plant", "green leaf"]
PRICE = settings.sam_usd_per_image


def plantseg_pairs(split: str | None, plants: list[str], diseases: list[str]) -> tuple[str, list[tuple[Path, Path]]]:
    ps = settings.datasets_dir / "PlantSeg"
    splits = [split] if split else ["val", "validation", "train"]
    for sp in splits:
        imgs, masks = ps / "images" / sp, ps / "annotations" / sp
        if sp == "test":
            sys.exit("Don't choose phrases on the test split: that's what the report uses.")
        if not (imgs.is_dir() and masks.is_dir()):
            continue
        meta_path = find_meta(ps)
        meta = read_meta(meta_path) if meta_path else {}
        out = []
        for p in sorted(imgs.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            m = next((masks / f"{p.stem}{e}" for e in (".png", ".jpg") if (masks / f"{p.stem}{e}").exists()), None)
            info = meta.get(p.stem, {})
            text = f"{info.get('plant', '')} {info.get('disease', '')} {p.stem}".lower()
            if m and (not plants or any(x.lower() in text for x in plants)) \
                    and (not diseases or any(x.lower() in text for x in diseases)):
                out.append((p, m))
        return sp, out
    sys.exit(f"No PlantSeg val/train split under {ps / 'images'}. Pass --split with the folder name.")


def ccmt_dev(n: int, seed: int) -> list[dict]:
    man = HERE / "data" / "images_manifest.csv"
    if not man.exists():
        return []
    rows = [r for r in csv.DictReader(man.open(newline="", encoding="utf-8"))
            if r["split"] == "dev" and not r["label"].endswith("/healthy")
            and (settings.datasets_dir / "CCMT" / r["path"]).exists()]
    rng = random.Random(seed)
    by_crop: dict[str, list[dict]] = {}
    for r in rows:
        by_crop.setdefault(r["crop"], []).append(r)
    picked = []
    for crop in sorted(by_crop):
        rng.shuffle(by_crop[crop])
        picked += by_crop[crop][: max(1, n // len(by_crop))]
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually make the calls")
    ap.add_argument("--split", help="PlantSeg folder for choosing (default: val, else train)")
    ap.add_argument("--plants", nargs="*", default=["corn", "maize"])
    ap.add_argument("--diseases", nargs="*", default=["leaf spot"])
    ap.add_argument("--n-lesion", type=int, default=8)
    ap.add_argument("--n-leaf", type=int, default=6)
    ap.add_argument("--lesion-phrases", nargs="*", default=LESION_PHRASES)
    ap.add_argument("--leaf-phrases", nargs="*", default=LEAF_PHRASES, help="{crop} becomes maize / cassava")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    split, pairs = plantseg_pairs(args.split, args.plants, args.diseases)
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.n_lesion]
    ccmt = ccmt_dev(args.n_leaf, args.seed)
    calls = len(pairs) * len(args.lesion_phrases) + len(ccmt) * len(args.leaf_phrases)
    print(f"Damage phrases: {len(args.lesion_phrases)} x {len(pairs)} PlantSeg '{split}' photos "
          f"({', '.join(args.plants)} / {', '.join(args.diseases)})")
    print(f"Leaf phrases:   {len(args.leaf_phrases)} x {len(ccmt)} CCMT dev photos")
    print(f"Up to {calls} SAM calls = ${calls * PRICE:.3f} (cached replies are free).")
    if not pairs:
        print("No PlantSeg photos matched; check --plants/--diseases (python bench/sam_eval.py --list shows them).")
    if not args.yes:
        print("Nothing sent. Add --yes to run it.")
        return
    if settings.cache_only:
        sys.exit("FIELDWISE_CACHE_ONLY=1 is on. In this window run:  $env:FIELDWISE_CACHE_ONLY=\"0\"")

    sam, spent, done = SamClient(), 0.0, 0
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    report: dict = {"plantseg_split": split, "lesion": {}, "leaf": {}}

    # ---- damage phrases, scored against PlantSeg lesion masks
    prepared = []
    for img, mask in pairs:
        photo = prepare(img.read_bytes())
        prepared.append((img.stem, photo, gt_mask(mask, (photo.width, photo.height))))
    for phrase in args.lesion_phrases:
        ious, errs, zero = [], [], 0
        for stem, photo, gt in prepared:
            try:
                masks, cost, _ = sam.segment(photo.jpeg, phrase)
            except SamUnavailable as e:
                stop_on(e, done)
            spent, done = spent + cost, done + 1
            pred, n, _ = rasterize(masks.raw_output, photo.width, photo.height)
            ious.append(iou(pred, gt))
            errs.append(100 * abs(int(pred.sum()) - int(gt.sum())) / gt.size)
            zero += n == 0
        report["lesion"][phrase] = {"mean_iou": round(float(np.mean(ious)), 4), "median_iou": round(float(np.median(ious)), 4),
                                    "median_abs_error_pts": round(float(np.median(errs)), 2), "zero_match": zero, "n": len(ious)}
        print(f"  damage {phrase!r:<16} mean IoU {np.mean(ious):.2f}  median {np.median(ious):.2f}  "
              f"area error {np.median(errs):5.1f} pts  found nothing on {zero}/{len(ious)}")

    # ---- leaf phrases on CCMT dev photos (overlays saved: CCMT is CC BY 4.0)
    ov_dir = out_dir / "sam_leaf_overlays"
    ov_dir.mkdir(exist_ok=True)
    for template in args.leaf_phrases:
        found, cover = 0, []
        for r in ccmt:
            phrase = template.replace("{crop}", r["crop"])
            photo = prepare((settings.datasets_dir / "CCMT" / r["path"]).read_bytes())
            try:
                masks, cost, _ = sam.segment(photo.jpeg, phrase)
            except SamUnavailable as e:
                stop_on(e, done)
            spent, done = spent + cost, done + 1
            leaf, n, _ = rasterize(masks.raw_output, photo.width, photo.height)
            found += n > 0
            cover.append(100 * float(leaf.mean()))
            png = base64.b64decode(overlay_png(photo.jpeg, leaf, np.zeros_like(leaf)))
            safe = template.replace("{crop}", "CROP").replace(" ", "_")
            (ov_dir / f"{r['id']}__{safe}.png").write_bytes(png)
        report["leaf"][template] = {"found_on": found, "n": len(ccmt),
                                    "median_cover_pct": round(statistics.median(cover), 1) if cover else None}
        print(f"  leaf   {template!r:<16} found a leaf on {found}/{len(ccmt)}  median cover "
              f"{statistics.median(cover) if cover else 0:5.1f}% of photo")

    best_lesion = max(report["lesion"].items(), key=lambda kv: (kv[1]["mean_iou"], -kv[1]["zero_match"]), default=None)
    best_leaf = max(report["leaf"].items(), key=lambda kv: (kv[1]["found_on"], kv[1]["median_cover_pct"] or 0), default=None)
    report.update(best_lesion=best_lesion[0] if best_lesion else None, best_leaf=best_leaf[0] if best_leaf else None,
                  api_cost_usd=round(spent, 4))
    (out_dir / "sam_phrases.json").write_text(json.dumps(report, indent=2))
    print(f"\nSpent ${spent:.4f}. Saved bench/results/sam_phrases.json.")
    if best_lesion:
        print(f"Best damage phrase on '{split}': {best_lesion[0]!r} (mean IoU {best_lesion[1]['mean_iou']:.2f}). "
              f"Report it once on test:\n  python bench\\sam_eval.py --plants {' '.join(args.plants)} "
              f"--diseases \"{args.diseases[0]}\" --phrase \"{best_lesion[0]}\" --n 15 --out bench\\results\\sam_eval_leafspot.csv")
    if best_leaf:
        print(f"Best leaf phrase: {best_leaf[0]!r}. Look at the overlays in bench\\results\\sam_leaf_overlays "
              "(dimmed = not leaf) before trusting it.")


if __name__ == "__main__":
    main()
