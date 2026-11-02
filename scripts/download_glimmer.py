"""Download a Muse Glimmer GGUF build into models/ (works on Windows, macOS and Linux).

    python scripts/download_glimmer.py                    # IQ3_XXS (11.55 GB) + vision projector
    python scripts/download_glimmer.py --quant IQ2_M      # smaller (10.66 GB) if IQ3_XXS runs out of memory
    python scripts/download_glimmer.py --no-vision        # text only (photo diagnosis won't work)

Uses the huggingface_hub library directly, so it doesn't matter whether the `hf`
command is on your PATH. Interrupted downloads resume when you run it again.

Source: bartowski/Muse-Glimmer-30B-GGUF, a community quantisation of Meta's
Muse Glimmer 30B (Apache 2.0). Check the hackathon rules allow community builds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "bartowski/Muse-Glimmer-30B-GGUF"
SIZES_GB = {"IQ2_M": 10.66, "IQ3_XXS": 11.55, "IQ3_XS": 12.32, "Q3_K_S": 12.79, "Q3_K_M": 13.96,
            "IQ4_XS": 15.44, "Q4_K_M": 17.31}   # as listed on the repo page, 27 Oct 2026
MMPROJ = "mmproj-Muse-Glimmer-30B-f16.gguf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="IQ3_XXS", choices=sorted(SIZES_GB))
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--dir", type=Path, default=ROOT / "models")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        sys.exit("Install the downloader first:  python -m pip install -U huggingface_hub")

    model_file = f"Muse-Glimmer-30B-{args.quant}.gguf"
    wanted = [model_file] + ([] if args.no_vision else [MMPROJ])

    available = set(list_repo_files(REPO))
    missing = [f for f in wanted if f not in available]
    if missing:
        ggufs = sorted(f for f in available if f.endswith(".gguf"))
        sys.exit(f"Not in {REPO}: {missing}\nFiles there now:\n  " + "\n  ".join(ggufs))

    args.dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading into {args.dir} (model ~{SIZES_GB[args.quant]} GB). Safe to stop and re-run; it resumes.")
    paths = [Path(hf_hub_download(REPO, f, local_dir=args.dir)) for f in wanted]
    for p in paths:
        print(f"  {p}  ({p.stat().st_size / 1e9:.2f} GB)")

    rel = lambda p: p.relative_to(ROOT) if p.is_relative_to(ROOT) else p  # noqa: E731
    print("\nPut these in your .env (the IQ3_XXS lines are already in .env.example):")
    print(f"  GLIMMER_GGUF={rel(paths[0]).as_posix()}")
    print(f"  GLIMMER_MMPROJ={rel(paths[1]).as_posix()}" if not args.no_vision else "  GLIMMER_MMPROJ=")
    print("Then start it:  .\\scripts\\start_glimmer.ps1   (Windows)   or   ./scripts/start_glimmer.sh")


if __name__ == "__main__":
    main()
