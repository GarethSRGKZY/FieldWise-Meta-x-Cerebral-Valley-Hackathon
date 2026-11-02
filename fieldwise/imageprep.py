"""Photo clean-up before any model sees it.

- Applies EXIF orientation first, so SAM's pixel coordinates match what the
  farmer saw (the SAM docs require the displayed orientation).
- Re-encodes the pixels only. All EXIF is dropped, including GPS coordinates
  that phones embed: a location leak that text redaction would never catch.
- Downscales the long side. Fewer image tokens means faster local inference
  and cheaper Spark calls.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps

GPS_IFD = 0x8825


@dataclass
class PreparedImage:
    jpeg: bytes
    width: int
    height: int
    had_exif: bool
    had_gps: bool


def prepare(raw: bytes, max_side: int = 1024, quality: int = 88) -> PreparedImage:
    img = Image.open(io.BytesIO(raw))
    exif = img.getexif()
    had_exif = len(exif) > 0
    had_gps = False
    try:
        had_gps = bool(exif.get_ifd(GPS_IFD))
    except (KeyError, ValueError, AttributeError):
        had_gps = False

    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)  # no exif= argument -> none written
    return PreparedImage(buf.getvalue(), img.width, img.height, had_exif, had_gps)
