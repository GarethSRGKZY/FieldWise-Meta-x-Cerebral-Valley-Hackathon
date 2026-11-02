"""Closed label set for photo diagnosis.

Classes follow the CCMT dataset (Ghana, CC BY 4.0, doi:10.17632/bwh3zbpkpv.1).
Each crop gets lettered options plus an explicit "unsure" option, so the model
answers with one token and we can read a real probability from its logprobs.

`sam_phrase` is the short noun phrase sent to SAM 3.1 for the damaged area.
Tune these on dev images only. `None` means no lesion to measure.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

UNSURE_LETTER = "Z"


@dataclass(frozen=True)
class Label:
    key: str          # stable id, e.g. "maize/fall_armyworm"
    crop: str
    name: str         # human-readable
    kind: str         # "disease" | "pest" | "healthy"
    sam_phrase: str | None
    aliases: tuple[str, ...] = ()


_RAW: dict[str, list[tuple[str, str, str, str | None, tuple[str, ...]]]] = {
    "maize": [
        ("healthy", "Healthy", "healthy", None, ()),
        ("fall_armyworm", "Fall armyworm damage", "pest", "leaf damage", ("fall armyworm", "faw", "spodoptera frugiperda")),
        ("grasshopper", "Grasshopper damage", "pest", "leaf damage", ("grasshoper", "grasshopper")),
        ("leaf_beetle", "Leaf beetle damage", "pest", "leaf damage", ("leaf beetle",)),
        ("leaf_blight", "Leaf blight", "disease", "brown lesion", ("northern leaf blight", "leaf blight")),
        ("leaf_spot", "Leaf spot", "disease", "leaf spot", ("gray leaf spot", "leaf spot")),
        ("streak_virus", "Maize streak virus", "disease", "yellow streak", ("maize streak", "streak virus", "msv")),
    ],
    "cassava": [
        ("healthy", "Healthy", "healthy", None, ()),
        ("bacterial_blight", "Cassava bacterial blight", "disease", "brown lesion", ("bacterial blight", "cbb")),
        ("brown_spot", "Brown leaf spot", "disease", "brown spot", ("brown spot", "brown leaf spot")),
        ("green_mite", "Cassava green mite", "pest", "yellow spot", ("green mite", "mite")),
        ("mosaic", "Cassava mosaic disease", "disease", "yellow patch", ("mosaic", "cmd", "cassava mosaic")),
    ],
    "tomato": [
        ("healthy", "Healthy", "healthy", None, ()),
        ("leaf_blight", "Leaf blight", "disease", "brown lesion", ("early blight", "late blight", "leaf blight")),
        ("leaf_curl", "Leaf curl", "disease", None, ("leaf curl", "tylcv", "yellow leaf curl")),
        ("septoria_leaf_spot", "Septoria leaf spot", "disease", "leaf spot", ("septoria",)),
        ("verticillium_wilt", "Verticillium wilt", "disease", "yellow patch", ("verticulium wilt", "verticillium")),
    ],
}

LABELS: dict[str, Label] = {}
for crop, rows in _RAW.items():
    for key, name, kind, phrase, aliases in rows:
        label = Label(f"{crop}/{key}", crop, name, kind, phrase, aliases)
        LABELS[label.key] = label

# Crops switched on in the app and benchmark. Maize and cassava have licensed
# FAO guides; tomato stays defined but off until a guide covering its classes
# is found. Override with FIELDWISE_CROPS=maize,cassava,tomato
_wanted = [c.strip().lower() for c in os.environ.get("FIELDWISE_CROPS", "maize,cassava").split(",") if c.strip()]
_unknown = [c for c in _wanted if c not in _RAW]
if _unknown:
    raise ValueError(f"FIELDWISE_CROPS has unknown crop(s) {_unknown}; defined: {list(_RAW)}")
CROPS = tuple(_wanted)
ALL_CROPS = tuple(_RAW.keys())


def options_for(crop: str) -> list[tuple[str, Label | None]]:
    """Lettered options for one crop; the last is always the unsure option."""
    crop = crop.lower()
    if crop not in _RAW:
        raise ValueError(f"Unknown crop {crop!r}; expected one of {CROPS}")
    labels = [LABELS[f"{crop}/{row[0]}"] for row in _RAW[crop]]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXY"
    opts: list[tuple[str, Label | None]] = [(letters[i], lab) for i, lab in enumerate(labels)]
    opts.append((UNSURE_LETTER, None))
    return opts


def normalise_class_name(raw: str) -> str:
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


def match_label(crop: str, raw_class: str) -> Label | None:
    """Map a dataset folder name (e.g. 'Fall Armyworm', 'verticulium wilt') to a Label."""
    want = normalise_class_name(raw_class)
    for (_, lab) in options_for(crop):
        if lab is None:
            continue
        cands = {lab.key.split("/", 1)[1]} | {normalise_class_name(a) for a in lab.aliases}
        if want in cands:
            return lab
    return None
