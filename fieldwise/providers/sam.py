"""SAM 3.1 on the Meta Model API: leaf and lesion masks -> % of leaf affected.

Per the docs: Responses API, model `sam-3.1`, one short noun phrase per
request, image as a base64 data URL, streamed special-token output parsed with
`meta_sam_parser`. Masks are box-local rasters; `bounds` place them in the
image. Coordinates are in the image's displayed orientation, so callers must
apply EXIF orientation before sending (imageprep.prepare does).

Cost: $2.50 per 1,000 images, so two prompts per photo is $0.005.
"""
from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field

import httpx
import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
)
from PIL import Image

from ..cache import DiskCache, request_key
from ..config import settings
from .glimmer import image_data_url
from .spark import key_refused, note_key_refused


class SamUnavailable(RuntimeError):
    pass


@dataclass
class MaskSet:
    phrase: str
    raw_output: str
    instances: int
    confidences: list[float] = field(default_factory=list)


@dataclass
class SeverityResult:
    percent_affected: float | None   # lesion area inside leaf / leaf area * 100
    leaf_found: bool
    lesion_phrase: str | None
    leaf_instances: int
    lesion_instances: int
    overlay_png_b64: str | None
    cost_usd: float
    latency_s: float
    cached: bool
    note: str = ""


# Bump when the way replies are read changes, so replies saved by an older reader are fetched again.
# v2 (30 Oct): read the final text from the done events; the live API sends no text deltas.
CACHE_VERSION = 2


def collect_stream_text(lines) -> str:
    """The output text of one SAM Responses stream (server-sent event lines).

    Per Meta's protocol (meta-sam/protocol/sam3.md, "Streaming and outcomes"): text deltas, if any,
    are concatenated exactly as sent (they can split anywhere, even inside a mask payload). The lane
    is finalized by `response.output_text.done` (`text`) or `response.content_part.done` with an
    `output_text` part (`part.text`); without deltas, that final text is the whole output. The live
    API sends only the done events. An empty text is a valid "no matches"."""
    deltas: list[str] = []
    final: str | None = None
    for line in lines:
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        event = json.loads(data)
        etype = event.get("type")
        part = event.get("part") or {}
        if etype == "response.output_text.delta" and isinstance(event.get("delta"), str):
            deltas.append(event["delta"])
        elif etype == "response.output_text.done" and isinstance(event.get("text"), str):
            final = event["text"] if final is None else final
        elif etype == "response.content_part.done" and part.get("type") == "output_text" and isinstance(part.get("text"), str):
            final = part["text"] if final is None else final
        elif etype in ("response.refusal.delta", "response.refusal.done") or (
                etype == "response.content_part.done" and part.get("type") == "refusal"):
            raise SamUnavailable(f"SAM refused the request: {str(event)[:300]}")
        elif etype in ("response.failed", "response.incomplete", "error"):
            raise SamUnavailable(f"SAM stream ended with {etype}: {str(event)[:300]}")
    return "".join(deltas) if deltas else (final or "")


def parse_sam_text(raw_output: str):
    parser = image_segmentation_format().create_parser()
    parser.push(raw_output if raw_output.endswith("\n") else raw_output + "\n")
    return parser.finish(CompletedOutcome()).result


def rasterize(raw_output: str, width: int, height: int) -> tuple[np.ndarray, int, list[float]]:
    """Union of all instance masks as a (height, width) bool array."""
    canvas = np.zeros((height, width), dtype=bool)
    if not raw_output.strip():
        return canvas, 0, []
    result = parse_sam_text(raw_output)
    count = 0
    confs: list[float] = []
    for rec in result.records:
        if not isinstance(rec, SegmentationMaskRecord):
            continue
        count += 1
        if rec.confidence is not None:
            confs.append(float(rec.confidence))
        raster = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8)
        raster = raster.reshape(rec.mask.height, rec.mask.width)
        left, top = int(round(rec.bounds.left)), int(round(rec.bounds.top))
        right, bottom = int(round(rec.bounds.right)), int(round(rec.bounds.bottom))
        bw, bh = max(right - left, 1), max(bottom - top, 1)
        scaled = Image.fromarray(raster * 255).resize((bw, bh), Image.NEAREST)
        block = np.asarray(scaled) > 127
        x0, y0 = max(left, 0), max(top, 0)
        x1, y1 = min(left + bw, width), min(top + bh, height)
        if x1 <= x0 or y1 <= y0:
            continue
        canvas[y0:y1, x0:x1] |= block[y0 - top:y1 - top, x0 - left:x1 - left]
    return canvas, count, confs


def overlay_png(jpeg: bytes, leaf: np.ndarray, lesion: np.ndarray) -> str:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    out = arr.copy()
    # dim everything outside the leaf, tint lesions inside it
    if leaf.any():
        out[~leaf] *= 0.45
    hit = lesion & leaf if leaf.any() else lesion
    out[hit] = out[hit] * 0.35 + np.array([230, 40, 40], dtype=np.float32) * 0.65
    buf = io.BytesIO()
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class SamClient:
    def __init__(self):
        self.cache = DiskCache(settings.cache_dir, "sam")
        self._http = httpx.Client(timeout=settings.sam_timeout_s)

    RETRY_DELAYS_S = (2.0, 5.0)          # 3 attempts in all

    def _post_with_retries(self, body: dict, headers: dict) -> str:
        """POST and read the stream. Network failures (timeouts, dropped connections, DNS) and HTTP
        429/5xx are retried; other HTTP errors (401/403 key problems, 400) are not, since retrying
        can't fix them. A retry after a read timeout may be billed twice ($0.0025)."""
        if key_refused():
            raise SamUnavailable(f"SAM HTTP 403: not sent, Meta refused the key earlier in this session ({key_refused()})")
        last = ""
        for attempt, delay in enumerate((*self.RETRY_DELAYS_S, None), start=1):
            try:
                with self._http.stream("POST", f"{settings.meta_base_url}/responses", json=body, headers=headers) as r:
                    if r.status_code == 200:
                        return collect_stream_text(r.iter_lines())
                    last = f"SAM HTTP {r.status_code}: {r.read()[:300]!r}"
                    if r.status_code in (401, 403):
                        note_key_refused(last)
                    if r.status_code != 429 and r.status_code < 500:
                        raise SamUnavailable(last)
            except httpx.HTTPError as e:
                last = f"SAM not reachable: {e}"
            if delay is None:
                break
            time.sleep(delay)
        raise SamUnavailable(f"{last} (after {len(self.RETRY_DELAYS_S) + 1} attempts)")

    def segment(self, jpeg: bytes, phrase: str) -> tuple[MaskSet, float, bool]:
        """Returns (masks, cost_usd, cached)."""
        body = {
            "model": settings.sam_model,
            "input": [{
                "type": "message", "role": "user",
                "content": [
                    {"type": "input_text", "text": phrase},
                    {"type": "input_image", "image_url": image_data_url(jpeg)},
                ],
            }],
            "stream": True,
            "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
        }
        key = request_key({"cache_version": CACHE_VERSION, "request": body})   # the version is never sent
        hit = self.cache.get(key)
        if hit is not None:
            return MaskSet(phrase, hit["raw"], hit["instances"], hit["conf"]), 0.0, True
        if settings.cache_only:
            raise SamUnavailable("Cache-only mode and this image was never segmented.")
        if not settings.meta_api_key:
            raise SamUnavailable("MODEL_API_KEY is not set.")

        headers = {
            "Authorization": f"Bearer {settings.meta_api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        raw = self._post_with_retries(body, headers)

        result = parse_sam_text(raw) if raw.strip() else None
        masks = [r for r in (result.records if result else []) if isinstance(r, SegmentationMaskRecord)]
        conf = [float(m.confidence) for m in masks if m.confidence is not None]
        self.cache.put(key, {"raw": raw, "instances": len(masks), "conf": conf})
        return MaskSet(phrase, raw, len(masks), conf), settings.sam_usd_per_image, False

    def severity(self, jpeg: bytes, lesion_phrase: str | None, leaf_phrase: str = "leaf",
                 method: str | None = None) -> SeverityResult:
        """% of the leaf that is damaged. SAM 3.1 finds the leaf; the damaged part comes from colour
        inside it (method "colour", the default) or from a SAM text prompt (method "sam")."""
        from ..lesions import colour_lesions
        method = method or settings.lesion_method
        t0 = time.perf_counter()
        w, h = Image.open(io.BytesIO(jpeg)).size
        leaf_set, c1, cached1 = self.segment(jpeg, leaf_phrase)
        leaf, n_leaf, _ = rasterize(leaf_set.raw_output, w, h)
        if not leaf.any():
            return SeverityResult(None, False, None, n_leaf, 0, overlay_png(jpeg, leaf, np.zeros_like(leaf)), c1,
                                  time.perf_counter() - t0, cached1,
                                  "No leaf found, so the damaged share can't be computed. Try a closer photo of one leaf.")
        if method == "colour":
            lesion = colour_lesions(jpeg, leaf, settings.lesion_hue_min, settings.lesion_sat_min)
            pct = 100.0 * float(lesion.sum()) / float(leaf.sum())
            return SeverityResult(pct, True, None, n_leaf, 0, overlay_png(jpeg, leaf, lesion), c1,
                                  time.perf_counter() - t0, cached1,
                                  "Leaf found by SAM 3.1; damaged tissue = parts of the leaf that aren't healthy green.")
        if not lesion_phrase:
            return SeverityResult(0.0, True, None, n_leaf, 0, overlay_png(jpeg, leaf, np.zeros_like(leaf)), c1,
                                  time.perf_counter() - t0, cached1,
                                  "No lesion phrase for this class (e.g. healthy or whole-plant symptoms).")
        les_set, c2, cached2 = self.segment(jpeg, lesion_phrase)
        lesion, n_les, _ = rasterize(les_set.raw_output, w, h)
        pct = 100.0 * float((lesion & leaf).sum()) / float(leaf.sum())
        return SeverityResult(pct, True, lesion_phrase, n_leaf, n_les, overlay_png(jpeg, leaf, lesion), c1 + c2,
                              time.perf_counter() - t0, cached1 and cached2, "")
