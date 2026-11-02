"""Validate SAM 3.1 lesion masks against hand-checked ground truth.

Default source: PlantSeg (Wei et al. 2024), in-the-wild plant disease images
with pixel masks of the diseased area. Licence: CC BY-NC-ND 4.0 (check the
copy you download). Non-commercial, no derivatives: we only report numbers
and never redistribute images or masks. Check the hackathon rules on
non-commercial data before relying on it.

    python bench/sam_eval.py --plants corn maize cassava --n 30
        # defaults: datasets/PlantSeg/images/test, datasets/PlantSeg/annotations/test,
        # datasets/PlantSeg/Metadatav2.csv. Pass --images/--masks/--meta if your copy differs.

You can also point --images/--masks at your own hand-labelled folder (same
file stem, mask PNG with nonzero = lesion). Cost: n x $0.0025.

Reports per image: lesion IoU and the error in "% of image that is lesion",
since PlantSeg masks lesions but not whole leaves. The leaf denominator in the
app's severity % is therefore NOT validated by this script; say so, or
hand-label ~15 leaf masks and pass --leaf-masks.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fieldwise.imageprep import prepare  # noqa: E402
from fieldwise.lesions import colour_lesions  # noqa: E402
from fieldwise.providers.sam import SamClient, SamUnavailable, rasterize  # noqa: E402

HERE = Path(__file__).resolve().parent


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def gt_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    m = Image.open(path).convert("L").resize(size, Image.NEAREST)
    return np.asarray(m) > 0


def stop_on(e: SamUnavailable, done: int) -> None:
    """Explain a SAM failure in plain words and stop, instead of a traceback."""
    from fieldwise.config import settings
    msg = str(e)
    if "Cache-only" in msg:
        sys.exit(f"Stopped after {done} image(s): FIELDWISE_CACHE_ONLY=1 is on (in .env), so this script only reuses "
                 "saved SAM results and never calls Meta, and this image was never segmented.\n"
                 "Only once Meta has confirmed your access is restored, turn it off for this one window and re-run:\n"
                 "  PowerShell:  $env:FIELDWISE_CACHE_ONLY=\"0\"; python bench\\sam_eval.py ...same options...\n"
                 "Leave .env as it is, so the demo still never calls the network.")
    if "HTTP 401" in msg or "HTTP 403" in msg:
        sys.exit(f"Stopped after {done} image(s): Meta still refuses the key ({msg[:160]}). "
                 "Access hasn't been restored; don't retry until Meta says it has.")
    sys.exit(f"Stopped after {done} image(s): {msg[:300]}")


def find_meta(root: Path) -> Path | None:
    """PlantSeg releases name the metadata file differently (Metadatav2.csv, metadata.csv, ...).
    Pick the CSV whose name mentions 'meta', else the only CSV with plant and disease columns."""
    csvs = sorted(root.rglob("*.csv")) if root.is_dir() else []
    named = [c for c in csvs if "meta" in c.name.lower()]
    if named:
        return named[0]
    for c in csvs:
        try:
            header = c.open(encoding="utf-8-sig").readline().lower()
        except OSError:
            continue
        if "plant" in header and ("disease" in header or "label" in header):
            return c
    return None


def read_meta(path: Path) -> dict[str, dict]:
    """PlantSeg metadata -> {image stem: {"plant": ..., "disease": ...}} (lower case). Column names are
    matched loosely ('plant', 'disease'/'label', 'name'/'image'/'file'), since releases differ."""
    out = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            keys = {k.lower(): k for k in row if k}
            pick = lambda *words: next((row[keys[k]] for k in keys if any(w in k for w in words)), "")  # noqa: E731
            name = pick("name", "image", "file")
            if name:
                out[Path(name).stem] = {"plant": str(pick("plant")).strip().lower(),
                                        "disease": str(pick("disease", "label")).strip().lower()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    from fieldwise.config import settings
    ps = settings.datasets_dir / "PlantSeg"
    ap.add_argument("--images", type=Path, default=ps / "images" / "test")
    ap.add_argument("--masks", type=Path, default=ps / "annotations" / "test")
    ap.add_argument("--leaf-masks", type=Path, help="optional hand-labelled whole-leaf masks")
    ap.add_argument("--meta", type=Path, default=ps / "Metadatav2.csv", help="PlantSeg metadata CSV (to filter by plant)")
    ap.add_argument("--plants", nargs="*", default=[], help="keep images whose plant matches, e.g. corn maize")
    ap.add_argument("--diseases", nargs="*", default=[], help="keep images whose disease matches, e.g. 'leaf spot'")
    ap.add_argument("--list", action="store_true", help="show plant/disease counts (with masks) and exit; no API calls")
    ap.add_argument("--method", choices=["colour", "sam"], default=None,
                    help="colour: SAM 3.1 leaf + off-colour pixels (the app's method, FIELDWISE_LESION_*); "
                         "sam: SAM 3.1 text prompt --phrase for the lesion. Default: FIELDWISE_LESION_METHOD")
    ap.add_argument("--phrase", default="leaf spot", help="SAM concept prompt for the diseased area (--method sam)")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "sam_eval.csv")
    ap.add_argument("--dry-run", action="store_true", help="check folders and count matching images; no API calls")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="earlier result CSVs whose photos must not be used again (e.g. after a decision was based on them)")
    args = ap.parse_args()
    for what, path in (("images", args.images), ("masks", args.masks)):
        if not path.is_dir():
            sys.exit(f"No {what} folder at {path}. Unzip PlantSeg under datasets/PlantSeg/ or pass --{what}.")
    needs_meta = args.plants or args.diseases or args.list
    if needs_meta and not (args.meta and args.meta.exists()):
        found = find_meta(ps)
        if found:
            print(f"Using metadata file {found}")
            args.meta = found
        else:
            csvs = sorted(ps.rglob("*.csv"))[:10]
            sys.exit(f"--plants/--diseases/--list need PlantSeg's metadata CSV, and none was found under {ps}."
                     + (" CSV files there: " + ", ".join(str(c) for c in csvs) + ". Pass the right one with --meta."
                        if csvs else " Is the metadata file in the download? Pass it with --meta."))

    stems = sorted(p.stem for p in args.images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    stems = [s for s in stems if any((args.masks / f"{s}{e}").exists() for e in (".png", ".jpg"))]
    if needs_meta:
        meta = read_meta(args.meta)
        if args.list:
            from collections import Counter
            have = set(stems)
            counts = Counter((m["plant"], m["disease"]) for stem, m in meta.items() if stem in have)
            print(f"{sum(counts.values())} images with masks in {args.images.name}. Plant / disease (count):")
            for (plant, disease), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {n:5d}  {plant} / {disease}")
            return
        def ok(stem: str) -> bool:
            m = meta.get(stem)
            if not m:
                return False
            return ((not args.plants or any(p.lower() in m["plant"] for p in args.plants))
                    and (not args.diseases or any(d.lower() in m["disease"] for d in args.diseases)))
        stems = [s for s in stems if ok(s)]
    if args.dry_run and not args.exclude:
        print(f"{len(stems)} images with a lesion mask{' for ' + ', '.join(args.plants) if args.plants else ''}; "
              f"a run of --n {min(args.n, len(stems))} would cost ${min(args.n, len(stems)) * 2.5 / 1000:.3f}.")
        return
    used = set()
    for f in args.exclude:
        if not f.exists():
            sys.exit(f"--exclude: no file {f}")
        with f.open(newline="", encoding="utf-8") as fh:
            used |= {r["image"] for r in csv.DictReader(fh)}
    if used:
        before = len(stems)
        stems = [s for s in stems if s not in used]
        print(f"Excluded {before - len(stems)} photos used before; {len(stems)} unused photos match.")
    if args.dry_run and used:
        print(f"A run of --n {min(args.n, len(stems))} would cost ${min(args.n, len(stems)) * 2.5 / 1000:.3f}.")
        return
    random.Random(args.seed).shuffle(stems)
    stems = stems[: args.n]
    if not stems:
        sys.exit("No image/mask pairs matched. Check --images, --masks and the plant filter.")

    method = args.method or settings.lesion_method
    leaf_phrase = settings.sam_leaf_phrase.replace("{crop}", "maize")
    hue_min, sat_min = settings.lesion_hue_min, settings.lesion_sat_min
    print(f"Method: {method}: " + (f"SAM 3.1 leaf ({leaf_phrase!r}) + colour (hue < {hue_min:g} or saturation < {sat_min:.2f})"
                                   if method == "colour" else f"SAM 3.1 text prompt {args.phrase!r}"))
    sam = SamClient()
    rows, spent = [], 0.0
    for stem in stems:
        img_path = next(p for p in args.images.iterdir() if p.stem == stem)
        photo = prepare(img_path.read_bytes())
        size = (photo.width, photo.height)
        mask_path = next(args.masks / f"{stem}{e}" for e in (".png", ".jpg") if (args.masks / f"{stem}{e}").exists())
        gt = gt_mask(mask_path, size)
        leaf = None
        try:
            if method == "colour":
                leaf_set, cost, cached = sam.segment(photo.jpeg, leaf_phrase)
                leaf, n_inst, conf = rasterize(leaf_set.raw_output, *size)
                pred = colour_lesions(photo.jpeg, leaf, hue_min, sat_min)
            else:
                masks, cost, cached = sam.segment(photo.jpeg, args.phrase)
                pred, n_inst, conf = rasterize(masks.raw_output, *size)
        except SamUnavailable as e:
            stop_on(e, len(rows))
        spent += cost
        area = gt.size
        row = {"image": stem, "instances": n_inst, "iou": round(iou(pred, gt), 4),
               "gt_lesion_pct_of_image": round(100 * gt.sum() / area, 2),
               "pred_lesion_pct_of_image": round(100 * pred.sum() / area, 2),
               "abs_error_pts": round(100 * abs(pred.sum() - gt.sum()) / area, 2),
               "mean_conf": round(float(np.mean(conf)), 3) if conf else "", "cached": cached}
        if leaf is not None:
            row["leaf_pct_of_image"] = round(100 * leaf.sum() / area, 2)
            if leaf.any():   # the app's number: damaged share of the leaf, vs PlantSeg's lesions inside the same leaf
                row["severity_pred_pct"] = round(100 * float((pred & leaf).sum()) / float(leaf.sum()), 2)
                row["severity_gt_pct"] = round(100 * float((gt & leaf).sum()) / float(leaf.sum()), 2)
                row["severity_abs_error_pts"] = round(abs(row["severity_pred_pct"] - row["severity_gt_pct"]), 2)
        if args.leaf_masks and (args.leaf_masks / f"{stem}.png").exists():
            leaf_gt = gt_mask(args.leaf_masks / f"{stem}.png", size)
            try:
                leaf_set, c2, _ = sam.segment(photo.jpeg, "leaf")
            except SamUnavailable as e:
                stop_on(e, len(rows))
            spent += c2
            leaf_pred, _, _ = rasterize(leaf_set.raw_output, *size)
            row["leaf_iou"] = round(iou(leaf_pred, leaf_gt), 4)
            sev_gt = 100 * (gt & leaf_gt).sum() / max(leaf_gt.sum(), 1)
            sev_pred = 100 * (pred & leaf_pred).sum() / max(leaf_pred.sum(), 1)
            row["severity_abs_error_pts"] = round(abs(sev_pred - sev_gt), 2)
        rows.append(row)
        print(f"  {stem}: IoU {row['iou']:.2f}, lesion {row['pred_lesion_pct_of_image']}% vs gt {row['gt_lesion_pct_of_image']}% of photo"
              + (f"; damaged share of leaf {row['severity_pred_pct']}% vs {row['severity_gt_pct']}%" if "severity_pred_pct" in row
                 else ("; no leaf found" if leaf is not None else "")))

    args.out.parent.mkdir(exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99))
        w.writeheader()
        w.writerows(rows)
    ious = [r["iou"] for r in rows]
    errs = [r["abs_error_pts"] for r in rows]
    summary = {"n": len(rows), "method": method,
               **({"leaf_phrase": leaf_phrase, "hue_min": hue_min, "sat_min": sat_min} if method == "colour" else {"phrase": args.phrase}), "mean_iou": float(np.mean(ious)), "median_iou": float(np.median(ious)),
               "median_abs_error_pts": float(np.median(errs)), "p90_abs_error_pts": float(np.percentile(errs, 90)),
               "zero_match_images": sum(r["instances"] == 0 for r in rows), "api_cost_usd": round(spent, 4)}
    if any("severity_abs_error_pts" in r for r in rows):
        sev = [r["severity_abs_error_pts"] for r in rows if "severity_abs_error_pts" in r]
        summary["severity_median_abs_error_pts"] = float(np.median(sev))
        summary["severity_p90_abs_error_pts"] = float(np.percentile(sev, 90))
        signed = [r["severity_pred_pct"] - r["severity_gt_pct"] for r in rows if "severity_pred_pct" in r]
        if signed:
            summary["severity_mean_signed_error_pts"] = float(np.mean(signed))   # + = over-estimates damage
            summary["severity_overestimated"] = sum(x > 0 for x in signed)
        summary["severity_n"] = len(sev)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
