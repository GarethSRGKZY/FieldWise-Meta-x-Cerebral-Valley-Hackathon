"""Replay Muse Glimmer's recorded benchmark outputs: a demo mode for a hub too small to run the model.

Glimmer 30B needs ~24 GB of memory. On a 16 GB laptop with a 4 GB GPU it reads ~1.4 prompt tokens/s
and times out, so the video can't show it answering live. With FIELDWISE_GLIMMER_REPLAY=1 the app
looks the question or photo up in the benchmark's Glimmer results (run on a Kaggle T4) and shows
that real output, confidence included. Retrieval, routing, redaction, the safety check, the queue
and the UI all still run live. Every replayed answer is labelled as such in the app.

Matching:
  questions: the benchmark question's text appears in what was typed (case and spacing ignored),
             with the same crop. So "My name is Kofi, 0244 123 456. <benchmark question>" still
             matches, which lets the demo show redaction.
  photos:    the same file (SHA-1, as in images_manifest.csv), else a near-identical picture
             (difference hash, for a copy a phone re-compressed), with the same crop.
Anything else is reported as "no recording", never made up.
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

from PIL import Image, ImageOps

from .config import ROOT, settings

BENCH = ROOT / "bench"
MIN_MATCH_CHARS = 20          # a recorded question shorter than this must match exactly
MAX_HASH_DISTANCE = 6         # of 64 bits; re-compressed copies of the same photo land at 0-4


def norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^\w'%/.-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .")


def dhash(data: bytes) -> int:
    """64-bit difference hash: survives re-compression and resizing, not cropping."""
    im = Image.open(io.BytesIO(data))
    im.draft("L", (64, 64))                       # fast JPEG decode at reduced size
    im = ImageOps.exif_transpose(im).convert("L").resize((9, 8), Image.LANCZOS)
    px = im.tobytes()                             # one byte per pixel in mode L
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def _last_by_id(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                out[r["id"]] = r          # the last good record wins, as in analyze.py
    return out


@dataclass
class Recording:
    id: str
    crop: str | None
    rec: dict
    item: dict


class GlimmerReplay:
    def __init__(self, results_dir: Path = BENCH / "results", questions: Path = BENCH / "data" / "questions.jsonl",
                 manifest: Path = BENCH / "data" / "images_manifest.csv", image_root: Path | None = None,
                 hash_cache: Path | None = None):
        self.image_root = image_root or settings.datasets_dir / "CCMT"
        self.hash_cache = hash_cache or settings.cache_dir / "replay_dhash.json"
        qa = _last_by_id(results_dir / "qa_local.jsonl")
        dx = _last_by_id(results_dir / "images_local.jsonl")
        self.questions: list[tuple[str, Recording]] = []
        if questions.exists():
            for line in questions.read_text("utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    if item["id"] in qa:
                        self.questions.append((norm(item["question"]), Recording(item["id"], item.get("crop"), qa[item["id"]], item)))
        self.questions.sort(key=lambda t: -len(t[0]))      # longest first: the most specific match wins
        self.photos: dict[str, Recording] = {}              # sha1 -> recording
        if manifest.exists():
            with manifest.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row["id"] in dx and row.get("sha1"):
                        self.photos[row["sha1"]] = Recording(row["id"], row["crop"], dx[row["id"]], row)
        self._hashes: dict[str, int] | None = None

    # ---------------------------------------------------------------- lookups
    def qa(self, question: str, crop: str | None) -> Recording | None:
        typed = norm(question)
        for text, r in self.questions:
            if r.crop != crop or not text:
                continue
            if typed == text or (len(text) >= MIN_MATCH_CHARS and text in typed):
                return r
        return None

    def dx(self, image_bytes: bytes, crop: str) -> Recording | None:
        r = self.photos.get(sha1(image_bytes).hexdigest())
        if r is None:
            r = self._nearest(image_bytes)
        return r if r and r.crop == crop else None

    def _nearest(self, image_bytes: bytes) -> Recording | None:
        try:
            h = dhash(image_bytes)
        except OSError:
            return None
        best, best_d = None, MAX_HASH_DISTANCE + 1
        for digest, hv in self._photo_hashes().items():
            d = bin(h ^ hv).count("1")
            if d < best_d:
                best, best_d = self.photos[digest], d
        return best

    def _photo_hashes(self) -> dict[str, int]:
        """dHash of every recorded photo, computed once from the CCMT files and cached on disk."""
        if self._hashes is not None:
            return self._hashes
        cached: dict[str, int] = {}
        try:
            cached = {k: int(v) for k, v in json.loads(self.hash_cache.read_text("utf-8")).items()}
        except (OSError, ValueError):
            pass
        for digest, r in self.photos.items():
            if digest not in cached:
                path = self.image_root / r.item["path"]
                try:
                    cached[digest] = dhash(path.read_bytes())
                except OSError:
                    continue                                 # dataset not on this machine: exact matches only
        try:
            self.hash_cache.parent.mkdir(parents=True, exist_ok=True)
            self.hash_cache.write_text(json.dumps({k: str(v) for k, v in cached.items()}), "utf-8")
        except OSError:
            pass
        self._hashes = {k: v for k, v in cached.items() if k in self.photos}
        return self._hashes

    # ----------------------------------------------------------------- status
    def health_detail(self) -> tuple[bool, str]:
        n_q, n_p = len(self.questions), len(self.photos)
        if not (n_q or n_p):
            return False, ("Replay mode is on (FIELDWISE_GLIMMER_REPLAY=1) but no recorded Glimmer results were found "
                           "in bench/results (qa_local.jsonl, images_local.jsonl).")
        return True, (f"Replay mode: showing Glimmer's recorded benchmark outputs for {n_q} questions and {n_p} photos. "
                      "Other questions get no Glimmer answer.")


def describe(rec: dict) -> dict:
    """What the UI shows about a replayed answer."""
    return {"source": "benchmark run on a Kaggle T4 GPU", "latency_s": rec.get("latency_s"), "recorded_at": rec.get("ts")}
