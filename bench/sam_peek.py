"""Look at what SAM 3.1 actually sent back, to tell "SAM found nothing" from "we didn't read its reply".

    python bench/sam_peek.py                 # the 15 newest saved SAM replies (no API calls, free)
    python bench/sam_peek.py --n 5 --chars 800
    python bench/sam_peek.py --probe
        # 2 photos x 2 phrases through the app's SAM client ($0.01): does SAM find anything at all?
        # (one PlantSeg corn photo and one CCMT maize photo; or pass your own with --images)
    python bench/sam_peek.py --live datasets\\PlantSeg\\images\\test\\corn_gray_leaf_spot_42.jpg --phrase leaf
        # ONE fresh call ($0.0025) that saves every raw stream event to bench/results/sam_stream_dump.txt

Every SAM reply is cached by request in data/cache/sam/*.json as the text we assembled from the
stream ("raw"), so an empty "raw" means our stream reading missed the content, while a long "raw"
with no mask records means the parser rejected it (see the diagnostics column).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fieldwise.config import settings  # noqa: E402
from fieldwise.providers.sam import parse_sam_text  # noqa: E402


def describe(raw: str) -> tuple[str, str]:
    if not raw.strip():
        return "nothing", ""
    try:
        res = parse_sam_text(raw)
    except Exception as e:  # noqa: BLE001
        return "parser error", f"{type(e).__name__}: {e}"
    kinds = Counter(type(r).__name__ for r in res.records)
    diags = "; ".join(f"{getattr(d, 'severity', '')} {getattr(d, 'code', '')} {getattr(d, 'message', d)}".strip()
                      for d in res.diagnostics)
    return (", ".join(f"{n} {k}" for k, n in kinds.items()) or "no records"), diags


def cached(n: int, chars: int) -> None:
    d = settings.cache_dir / "sam"
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:n] if d.exists() else []
    if not files:
        sys.exit(f"No saved SAM replies in {d}.")
    empty = 0
    for f in files:
        hit = json.loads(f.read_text("utf-8"))
        raw = hit.get("raw") or ""
        empty += not raw.strip()
        kinds, diags = describe(raw)
        when = datetime.fromtimestamp(f.stat().st_mtime).strftime("%d %b %H:%M")
        print(f"{when}  {f.stem[:10]}  raw {len(raw):6d} chars  saved instances {hit.get('instances')}  parsed: {kinds}")
        if diags:
            print(f"      diagnostics: {diags[:chars]}")
        if raw.strip():
            print(f"      start: {raw[:chars]!r}")
            if len(raw) > chars:
                print(f"      end:   {raw[-min(chars, 200):]!r}")
    print(f"\n{empty} of {len(files)} replies were empty."
          + (" Our stream reader likely missed the content: run once with --live to capture the raw events." if empty else ""))


def live(path: Path, phrase: str) -> None:
    import httpx

    from fieldwise.imageprep import prepare
    from fieldwise.providers.glimmer import image_data_url
    if settings.cache_only:
        sys.exit("FIELDWISE_CACHE_ONLY=1 is on; --live makes one paid call. Turn it off for this window first.")
    if not settings.meta_api_key:
        sys.exit("MODEL_API_KEY is not set.")
    jpeg = prepare(path.read_bytes()).jpeg
    body = {"model": settings.sam_model,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": phrase},
                {"type": "input_image", "image_url": image_data_url(jpeg)}]}],
            "stream": True, "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"}}
    headers = {"Authorization": f"Bearer {settings.meta_api_key}", "Content-Type": "application/json",
               "Accept": "text/event-stream"}
    out = HERE / "results" / "sam_stream_dump.txt"
    out.parent.mkdir(exist_ok=True)
    types: Counter = Counter()
    with httpx.Client(timeout=120) as c, c.stream("POST", f"{settings.meta_base_url}/responses", json=body, headers=headers) as r, \
            out.open("w", encoding="utf-8") as fh:
        print(f"HTTP {r.status_code}  content-type {r.headers.get('content-type')}")
        if r.status_code != 200:
            sys.exit(r.read()[:500].decode("utf-8", "replace"))
        for line in r.iter_lines():
            fh.write(line + "\n")
            if line.startswith("event:"):
                types[line[6:].strip()] += 1
            elif line.startswith("data: ") and line[6:] != "[DONE]":
                try:
                    types["data.type=" + str(json.loads(line[6:]).get("type"))] += 1
                except json.JSONDecodeError:
                    types["data (not JSON)"] += 1
    print("Event types in the stream:")
    for k, n in types.most_common():
        print(f"  {n:5d}  {k}")
    print(f"Full stream saved to {out}. Send me the counts above and the first ~40 lines of that file "
          "(it holds no key; it does hold mask data, which is just numbers).")


def probe_images() -> list[Path]:
    """One PlantSeg corn test photo and one CCMT maize photo from the benchmark manifest."""
    import csv
    out = []
    ps = settings.datasets_dir / "PlantSeg" / "images" / "test"
    corn = sorted(p for p in ps.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "corn" in p.name.lower()) if ps.is_dir() else []
    if corn:
        out.append(corn[0])
    man = HERE / "data" / "images_manifest.csv"
    if man.exists():
        for row in csv.DictReader(man.open(newline="", encoding="utf-8")):
            if row["crop"] == "maize" and not row["label"].endswith("/healthy"):
                path = settings.datasets_dir / "CCMT" / row["path"]
                if path.exists():
                    out.append(path)
                    break
    return out


def probe(images: list[Path], phrases: list[str]) -> None:
    import numpy as np

    from fieldwise.imageprep import prepare
    from fieldwise.providers.sam import SamClient, SamUnavailable, rasterize
    images = images or probe_images()
    if not images:
        sys.exit("No photos found; pass some with --images.")
    if settings.cache_only:
        sys.exit(f"FIELDWISE_CACHE_ONLY=1 is on; --probe makes up to {len(images) * len(phrases)} paid calls. "
                 "Turn it off for this window first.")
    sam, spent = SamClient(), 0.0
    print(f"{len(images)} photo(s) x {len(phrases)} phrase(s), up to ${len(images) * len(phrases) * 0.0025:.4f}\n")
    for img in images:
        photo = prepare(img.read_bytes())
        for phrase in phrases:
            try:
                masks, cost, cached = sam.segment(photo.jpeg, phrase)
            except SamUnavailable as e:
                sys.exit(f"{img.name} / {phrase!r}: {e}")
            spent += cost
            area, n, conf = rasterize(masks.raw_output, photo.width, photo.height)
            kinds, diags = describe(masks.raw_output)
            print(f"{img.name[:40]:<40} {phrase!r:<14} {n} object(s), {100 * area.mean():5.1f}% of photo"
                  + (f", confidence {min(conf):.2f}-{max(conf):.2f}" if conf else "") + ("  (saved reply)" if cached else ""))
            if diags:
                print(f"      diagnostics: {diags[:300]}")
    print(f"\nSpent ${spent:.4f}. If 'leaf' finds nothing on these, send me this output.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--chars", type=int, default=300)
    ap.add_argument("--live", type=Path, help="one paid call with this photo, saving every raw stream event")
    ap.add_argument("--phrase", default="leaf")
    ap.add_argument("--probe", action="store_true", help="paid check: does SAM find leaves / lesions at all?")
    ap.add_argument("--images", type=Path, nargs="*", default=[], help="photos for --probe")
    ap.add_argument("--phrases", nargs="*", default=["leaf", "leaf spot"], help="phrases for --probe")
    args = ap.parse_args()
    if args.probe:
        probe(args.images, args.phrases)
    elif args.live:
        live(args.live, args.phrase)
    else:
        cached(args.n, args.chars)


if __name__ == "__main__":
    main()
