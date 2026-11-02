"""Tune the colour cut-offs for damaged tissue inside SAM 3.1's leaf mask, on PlantSeg VAL photos only.

    python bench/lesion_tune.py              # plan and cost; no calls
    python bench/lesion_tune.py --yes        # ~20 SAM "leaf" calls ($0.05; cached ones are free)

For each photo: SAM 3.1 outlines the leaf (one call), then every (hue, saturation) cut-off pair on a
small grid marks off-colour pixels inside it (free, local). The pair with the smallest median error in
"% of the leaf damaged" (the number the app shows; PlantSeg's lesions inside the same leaf as truth)
wins; --objective iou picks by lesion overlap instead (v1, 30 Oct: it over-called damage on test).
Then report it ONCE on test photos not used before:

    python bench/sam_eval.py --method colour --plants corn maize --diseases "leaf spot" --n 15 \
        --exclude bench/results/sam_eval_leafspot_colour.csv --out bench/results/sam_eval_leafspot_colour_v2.csv

PlantSeg is CC BY-NC-ND: numbers only, nothing derived from its photos is saved.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from fieldwise.config import settings  # noqa: E402
from fieldwise.imageprep import prepare  # noqa: E402
from fieldwise.lesions import colour_lesions, hsv  # noqa: E402
from fieldwise.providers.sam import SamClient, SamUnavailable, rasterize  # noqa: E402
from sam_eval import gt_mask, iou, stop_on  # noqa: E402
from sam_phrases import plantseg_pairs  # noqa: E402

HUES = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
SATS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--split", help="PlantSeg folder for tuning (default: val, else train; never test)")
    ap.add_argument("--plants", nargs="*", default=["corn", "maize"])
    ap.add_argument("--diseases", nargs="*", default=["leaf spot"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--objective", choices=["severity", "iou"], default="severity")
    args = ap.parse_args()

    split, pairs = plantseg_pairs(args.split, args.plants, args.diseases)
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.n]
    leaf_phrase = settings.sam_leaf_phrase.replace("{crop}", "maize")
    print(f"{len(pairs)} PlantSeg '{split}' photos ({', '.join(args.plants)} / {', '.join(args.diseases)}); "
          f"SAM leaf phrase {leaf_phrase!r}; grid {len(HUES)} hues x {len(SATS)} saturations.")
    print(f"Up to {len(pairs)} SAM calls = ${len(pairs) * settings.sam_usd_per_image:.3f} (cached replies are free).")
    if not args.yes:
        print("Nothing sent. Add --yes to run it.")
        return
    if settings.cache_only:
        sys.exit("FIELDWISE_CACHE_ONLY=1 is on. In this window run:  $env:FIELDWISE_CACHE_ONLY=\"0\"")

    sam, spent, items, no_leaf = SamClient(), 0.0, [], 0
    for img, mask in pairs:
        photo = prepare(img.read_bytes())
        try:
            leaf_set, cost, _ = sam.segment(photo.jpeg, leaf_phrase)
        except SamUnavailable as e:
            stop_on(e, len(items))
        spent += cost
        leaf, n, _ = rasterize(leaf_set.raw_output, photo.width, photo.height)
        gt = gt_mask(mask, (photo.width, photo.height))
        if not leaf.any():
            no_leaf += 1
        items.append((photo, leaf, gt, hsv(photo.jpeg)))
        print(f"  {img.stem}: leaf {100 * leaf.mean():.0f}% of photo, true lesion {100 * gt.mean():.1f}% of photo")

    rows = []
    for hue in HUES:
        for sat in SATS:
            ious, signed = [], []
            for photo, leaf, gt, hsv_ in items:
                pred = colour_lesions(photo.jpeg, leaf, hue, sat, _hsv=hsv_)
                ious.append(iou(pred, gt))
                if leaf.any():
                    signed.append(100 * (float((pred & leaf).sum()) - float((gt & leaf).sum())) / float(leaf.sum()))
            rows.append({"hue_min": hue, "sat_min": sat, "mean_iou": float(np.mean(ious)),
                         "median_iou": float(np.median(ious)),
                         "median_severity_error_pts": float(np.median(np.abs(signed))) if signed else None,
                         "mean_signed_severity_error_pts": float(np.mean(signed)) if signed else None})
    if args.objective == "severity":
        rows.sort(key=lambda r: (r["median_severity_error_pts"] if r["median_severity_error_pts"] is not None else 1e9,
                                 -r["mean_iou"]))
    else:
        rows.sort(key=lambda r: -r["mean_iou"])
    best = rows[0]
    print(f"\nSAM found a leaf on {len(items) - no_leaf}/{len(items)} photos. Top cut-offs on '{split}' "
          f"(objective: {args.objective}):")
    for r in rows[:5]:
        print(f"  hue < {r['hue_min']:>2} or saturation < {r['sat_min']:.2f}:  damaged-share error {r['median_severity_error_pts']:.1f} pts "
              f"(median; mean bias {r['mean_signed_severity_error_pts']:+.1f})  mean IoU {r['mean_iou']:.2f}")
    if best["hue_min"] in (HUES[0], HUES[-1]) or best["sat_min"] in (SATS[0], SATS[-1]):
        print("  Note: the best pair is at the edge of the grid, so the true optimum may lie outside it. Say so in the write-up.")
    baseline = None
    pj = HERE / "results" / "sam_phrases.json"
    if pj.exists():
        lesion = json.loads(pj.read_text()).get("lesion") or {}
        if lesion:
            baseline = max(lesion.items(), key=lambda kv: kv[1]["mean_iou"])
            print(f"  (best SAM text prompt on val: {baseline[0]!r}, mean IoU {baseline[1]['mean_iou']:.2f})")
    out = {"objective": args.objective, "split": split, "n": len(items), "leaf_phrase": leaf_phrase, "leaf_found": len(items) - no_leaf,
           "best": best, "top5": rows[:5], "grid": {"hues": HUES, "sats": SATS},
           "sam_text_prompt_baseline": baseline, "api_cost_usd": round(spent, 4)}
    (HERE / "results" / f"lesion_tune_{args.objective}.json").write_text(json.dumps(out, indent=2))
    print(f"\nSpent ${spent:.4f}. Saved bench/results/lesion_tune_{args.objective}.json. Put these in .env:\n"
          f"  FIELDWISE_LESION_METHOD=colour\n  FIELDWISE_LESION_HUE_MIN={best['hue_min']}\n"
          f"  FIELDWISE_LESION_SAT_MIN={best['sat_min']:.2f}\n"
          "Then report once on test photos not used before:\n  python bench\\sam_eval.py --method colour --plants corn maize "
          "--diseases \"leaf spot\" --n 15 --exclude bench\\results\\sam_eval_leafspot_colour.csv "
          "--out bench\\results\\sam_eval_leafspot_colour_v2.csv")


if __name__ == "__main__":
    main()
