"""Damaged-area estimate: SAM 3.1 finds the leaf, colour finds the damaged tissue inside it."""
import io

import numpy as np
from PIL import Image

from fieldwise.config import settings
from fieldwise.lesions import colour_lesions
from fieldwise.providers import sam as sam_mod
from fieldwise.providers.sam import MaskSet, SamClient

from helpers import sam_line


def leaf_photo() -> bytes:
    """100x80: green leaf on the left 60 columns, soil on the right; patches inside the leaf."""
    a = np.zeros((80, 100, 3), np.uint8)
    a[:, :] = (120, 90, 60)                 # soil (brown, outside the leaf)
    a[:, :60] = (50, 140, 50)               # healthy green
    a[10:20, 10:20] = (150, 110, 60)        # tan/brown lesion
    a[30:40, 10:20] = (140, 140, 135)       # gray lesion (gray leaf spot)
    a[50:60, 10:20] = (200, 190, 40)        # yellow (streak / mosaic)
    a[10:20, 40:50] = (10, 25, 10)          # deep shadow: not judged
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")     # PNG: exact colours
    return buf.getvalue()


def test_off_colour_tissue_inside_the_leaf_only():
    leaf = np.zeros((80, 100), bool)
    leaf[:, :60] = True
    les = colour_lesions(leaf_photo(), leaf, hue_min=70, sat_min=0.2)
    assert les[10:20, 10:20].all() and les[30:40, 10:20].all() and les[50:60, 10:20].all()
    assert not les[10:20, 40:50].any()                 # shadow
    assert not les[65:80, 30:60].any()                 # healthy green
    assert not les[:, 60:].any()                       # brown soil outside the leaf is never counted
    assert les.sum() == 300


def test_severity_uses_sam_leaf_and_colour(monkeypatch):
    jpeg = leaf_photo()
    leaf_line = sam_line([(0, 0, 59, 79, [[1]])], 100, 80)
    calls = []

    def fake_segment(self, img, phrase):
        calls.append(phrase)
        return MaskSet(phrase, leaf_line, 1, [0.9]), 0.0025, False

    monkeypatch.setattr(SamClient, "segment", fake_segment)
    object.__setattr__(settings, "lesion_hue_min", 70.0)
    object.__setattr__(settings, "lesion_sat_min", 0.2)
    res = SamClient().severity(jpeg, "leaf spot", leaf_phrase="leaf", method="colour")
    assert calls == ["leaf"]                            # one SAM call: the leaf
    assert abs(res.percent_affected - 100 * 300 / 4800) < 0.01
    assert res.overlay_png_b64 and res.cost_usd == 0.0025


def test_no_leaf_means_no_number(monkeypatch):
    monkeypatch.setattr(SamClient, "segment", lambda self, img, phrase: (MaskSet(phrase, "", 0, []), 0.0025, False))
    res = SamClient().severity(leaf_photo(), "leaf spot", method="colour")
    assert res.percent_affected is None and not res.leaf_found and "No leaf" in res.note


def test_pest_damage_is_not_measured(tmp_path):
    from fieldwise.pipeline import Fieldwise
    from fieldwise.queue import EscalationQueue
    from fieldwise.retrieval import GuideIndex
    from helpers import FAKE_CHUNKS
    fw = Fieldwise(index=GuideIndex(FAKE_CHUNKS), queue=EscalationQueue(tmp_path / "q.sqlite3"))
    out = fw.severity(leaf_photo(), "maize/fall_armyworm")
    assert out["available"] is False and "insect" in out["note"]
