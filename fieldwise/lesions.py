"""Damaged tissue inside a leaf, by colour. SAM 3.1 finds the leaf; this marks what isn't healthy green.

Why: on 30 Oct, SAM 3.1 text prompts for lesions ("leaf spot", "lesion", "brown patch" and 5 more)
overlapped PlantSeg's lesion masks by at most 0.10 IoU on val photos, while "leaf" found the leaf on
every CCMT dev photo. Counting off-colour pixels inside a leaf mask is the usual way plant-disease
severity is measured from photos, so the leaf comes from SAM and the lesion from colour.

A pixel inside the leaf counts as damaged when it is bright enough to judge (not deep shadow) and
either its hue is outside the green band (yellow, tan, brown, red) or it has little colour at all
(gray or bleached tissue). The two cut-offs are tuned on PlantSeg val photos by bench/lesion_tune.py
and set in .env; they are reported with the test result.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

GREEN_HUE_MAX = 190.0     # hues above this (cyan, blue, purple, red) are not leaf green either


def hsv(jpeg: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue in degrees (0-360), saturation and value in 0-1, as (height, width) arrays."""
    arr = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB").convert("HSV"), dtype=np.float32)
    return arr[..., 0] * (360.0 / 255.0), arr[..., 1] / 255.0, arr[..., 2] / 255.0


def colour_lesions(jpeg: bytes, leaf: np.ndarray, hue_min: float, sat_min: float, val_min: float = 0.15,
                   _hsv: tuple | None = None) -> np.ndarray:
    """Pixels inside `leaf` that aren't healthy green.

    hue_min: hues below this (degrees) are not green: yellow is ~50-60, tan/brown ~20-40, green ~80-150.
    sat_min: saturation below this is gray or bleached, whatever the hue.
    val_min: darker pixels are shadow and left out."""
    h, s, v = _hsv if _hsv is not None else hsv(jpeg)
    if h.shape != leaf.shape:
        raise ValueError(f"leaf mask {leaf.shape} doesn't match the photo {h.shape}")
    off_colour = (h < hue_min) | (h > GREEN_HUE_MAX) | (s < sat_min)
    return leaf & (v >= val_min) & off_colour
