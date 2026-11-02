import io

import numpy as np
import pytest
from PIL import Image

from fieldwise import redact
from fieldwise.confidence import choice_from_logprobs, extract_answer_text
from fieldwise.imageprep import prepare
from fieldwise.labels import match_label, options_for
from fieldwise.providers.sam import rasterize
from fieldwise.retrieval import GuideIndex, chunk_pages
from fieldwise.router import Route, decide, qa_confidence
from fieldwise.safety import PPE_REMINDER, check

from helpers import FAKE_CHUNKS, sam_line


# ---------------------------------------------------------------- redaction
@pytest.mark.parametrize("text,kind,secret", [
    ("Hello, my name is Kofi Mensah and my maize is sick", "NAME", "Kofi Mensah"),
    ("This is Puan Siti Aminah from the cooperative", "NAME", "Siti Aminah"),
    ("Call me on +233 24 123 4567 please", "PHONE", "+233 24 123 4567"),
    ("my number 012-345 6789", "PHONE", "012-345 6789"),
    ("reach me at 0241234567", "PHONE", "0241234567"),
    ("email kofi.m@example.com", "EMAIL", "kofi.m@example.com"),
    ("the field is at 6.6885, -1.6244", "GPS", "6.6885, -1.6244"),
    ("IC 850101-14-5678", "ID", "850101-14-5678"),
    ("I farm near Ejura in the Ashanti Region", "LOCATION", "Ejura"),
    ("We live in Kampung Sungai Buloh", "LOCATION", "Kampung Sungai Buloh"),
    ("my neighbour Ama Owusu has it too", "NAME", "Ama Owusu"),
])
def test_redacts_personal_details(text, kind, secret):
    out, spans = redact.redact_text(text)
    assert secret not in out, out
    assert any(s.kind == kind for s in spans), spans


@pytest.mark.parametrize("text,secret", [
    ("my name is kofi mensah and my maize is sick", "kofi mensah"),
    ("this is kwame from Kumasi", "kwame"),
    ("hello, I'm kojo, the leaves are curling", "kojo"),
    ("im yaw, call me back", "yaw"),
    ("ama here, my cassava is yellow", "ama"),
    ("good morning, efua here from Cape Coast", "efua"),
    ("akosua from Sunyani here", "akosua"),
    ("Ama here, the okra is wilting", "Ama"),
])
def test_lowercase_and_opening_names(text, secret):
    out, spans = redact.redact_text(text)
    assert secret not in out, out
    assert any(s.kind == "NAME" for s in spans), spans


@pytest.mark.parametrize("text", [
    "i am worried about my maize",
    "im farming cassava near the river",
    "i'm a farmer and my leaves are yellow",
    "this is very bad, the armyworm is everywhere",
    "this is maize streak, i think",
    "it's yellow on the edges",
    "call me back tomorrow please",
    "maize here is dying after the rain",
    "everyone here has fall armyworm",
    "hello, farmers from our village are worried",
    "my name is not important, the cassava is sick",
    "i am asking about spraying",
])
def test_ordinary_words_after_triggers_are_kept(text):
    out, spans = redact.redact_text(text)
    assert out == text, spans


def test_name_and_place_are_separate_and_lowercase_words_kept():
    out, spans = redact.redact_text("My name is Kofi Mensah from Ejura and my maize is sick")
    assert out == "My name is [NAME] from [LOCATION] and my maize is sick"
    assert [s.kind for s in spans] == ["NAME", "LOCATION"]


@pytest.mark.parametrize("text", [
    "Holes in Maize leaves since March",
    "Mix 10 ml per 15 litres and spray in the morning",
    "I am worried about my tomato plants",
    "Spots appeared on 12/10/2026 after rain",
    "Planted 2500 plants per hectare",
])
def test_leaves_agronomy_alone(text):
    out, spans = redact.redact_text(text)
    assert out == text, spans


# ------------------------------------------------------------------- photos
def test_photo_gps_is_stripped_and_orientation_applied():
    img = Image.new("RGB", (400, 200), (10, 120, 30))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90
    gps = exif.get_ifd(0x8825)
    gps[2] = (6.0, 41.0, 18.0)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    prepared = prepare(buf.getvalue(), max_side=300)
    assert prepared.had_gps and prepared.had_exif
    out = Image.open(io.BytesIO(prepared.jpeg))
    assert len(out.getexif()) == 0
    assert out.size == (150, 300)  # rotated to portrait, then downscaled


# ---------------------------------------------------------------- retrieval
def test_bm25_finds_relevant_chunk_and_reports_coverage():
    idx = GuideIndex(FAKE_CHUNKS)
    res = idx.search("how do I scout maize for fall armyworm", k=2)
    assert res.passages[0].chunk_id == "t1#0"
    assert res.coverage > 0.5
    off = idx.search("banana weevil traps", k=2)
    assert off.coverage < 0.34


def test_chunking_keeps_pages():
    pages = [(1, "word " * 500), (2, "other " * 30)]
    chunks = chunk_pages(pages, words_per_chunk=200, overlap=20)
    assert {p for p, _ in chunks} == {1, 2}


# ------------------------------------------------------------------- safety
def test_unsupported_dose_is_removed_and_supported_kept():
    idx = GuideIndex(FAKE_CHUNKS)
    passages = idx.search("Product-A dose fall armyworm maize whorl", k=3).passages
    s_dose = next(i for i, p in enumerate(passages, 1) if p.chunk_id == "t1#1")
    answer = (f"Apply Product-A at 10 ml per 15 litres into the whorl [S{s_dose}]. "
              f"Repeat after 7 days [S{s_dose}]. Mix 50 ml per knapsack for best results.")
    rep = check(answer, passages)
    assert "10 ml per 15 litres" in rep.text
    assert "7 days" not in rep.text and "50 ml" not in rep.text
    assert len(rep.removed_sentences) == 2
    assert rep.chemical and PPE_REMINDER in rep.text


def test_invalid_citation_flagged():
    rep = check("Scout twice a week [S9].", [])
    assert rep.invalid_citations == [9]


# --------------------------------------------------------------- confidence
def test_letter_probability_from_logprobs():
    toks = [{"token": "ANSWER", "logprob": 0}, {"token": ":", "logprob": 0},
            {"token": " B", "logprob": -0.2, "top_logprobs": [
                {"token": " B", "logprob": -0.2}, {"token": " E", "logprob": -1.8}, {"token": " the", "logprob": -3}]}]
    c = choice_from_logprobs(toks, "ANSWER:", ["A", "B", "E", "Z"], "ANSWER: B")
    assert c.source == "logprobs" and c.chosen == "B"
    assert 0.75 < c.p_chosen < 0.9


def test_falls_back_to_text_without_logprobs():
    c = choice_from_logprobs(None, "SUPPORTED:", ["YES", "NO"], "ANSWER: blah\nSUPPORTED: NO")
    assert c.source == "text" and c.chosen == "NO"
    assert extract_answer_text("ANSWER: Scout fields.\nSUPPORTED: YES") == "Scout fields."


# ------------------------------------------------------------------ routing
def test_routing_rule():
    assert decide(0.9, 0.7, online=False) == Route.LOCAL
    assert decide(0.5, 0.7, online=True) == Route.CLOUD
    assert decide(0.5, 0.7, online=False) == Route.QUEUED
    assert qa_confidence(0.9, 0.4, True, True) == pytest.approx(0.45)
    assert qa_confidence(0.9, 1.0, True, False) == 0.0


# --------------------------------------------------------------------- labels
def test_labels_map_dataset_folders():
    assert match_label("maize", "Fall Armyworm").key == "maize/fall_armyworm"
    assert match_label("tomato", "verticulium wilt").key == "tomato/verticillium_wilt"
    assert options_for("cassava")[-1] == ("Z", None)


# ------------------------------------------------------------------------ SAM
def test_sam_masks_rasterise_into_image_space():
    # 100x80 image. Leaf: whole left half (box 0..49 x 0..79), mask all ones at 20x16.
    leaf_rows = [[1] * 20 for _ in range(16)]
    # Lesion: box 10..19 x 10..19, mask all ones at 5x5 -> 100 px
    lesion_rows = [[1] * 5 for _ in range(5)]
    leaf, n_leaf, conf = rasterize(sam_line([(0, 0, 49, 79, leaf_rows)], 100, 80), 100, 80)
    lesion, n_les, _ = rasterize(sam_line([(10, 10, 19, 19, lesion_rows)], 100, 80), 100, 80)
    assert n_leaf == 1 and n_les == 1 and conf == [0.9]
    assert leaf.sum() == 50 * 80
    assert lesion.sum() == 100
    pct = 100 * (lesion & leaf).sum() / leaf.sum()
    assert pct == pytest.approx(2.5)


def test_sam_zero_matches_is_empty_not_error():
    m, n, _ = rasterize("", 10, 10)
    assert n == 0 and not m.any()


def test_default_crops_are_maize_and_cassava():
    from fieldwise.labels import ALL_CROPS, CROPS, options_for
    assert CROPS == ("maize", "cassava")
    assert "tomato" in ALL_CROPS                      # still defined, just switched off
    assert len(options_for("maize")) + len(options_for("cassava")) - 2 == 12   # 12 classes + 2 "unsure"


def test_dotenv_loader_handles_comments_quotes_crlf_and_precedence(tmp_path, monkeypatch):
    from fieldwise.config import _load_dotenv
    env = tmp_path / ".env"
    env.write_bytes(b'# comment\r\nexport FW_A=one\r\nFW_B="two # not a comment"\r\nFW_C=3   # inline\r\n'
                    b'FW_C=4\r\nFW_D=from-file\r\n')
    for k in ("FW_A", "FW_B", "FW_C"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FW_D", "from-shell")
    _load_dotenv(env)
    import os
    assert os.environ["FW_A"] == "one"
    assert os.environ["FW_B"] == "two # not a comment"
    assert os.environ["FW_C"] == "4"                 # later line wins
    assert os.environ["FW_D"] == "from-shell"        # real environment wins
    for k in ("FW_A", "FW_B", "FW_C"):
        monkeypatch.delenv(k, raising=False)


def test_glimmer_health_explains_why_it_is_not_available():
    import httpx
    from fieldwise.providers.glimmer import GlimmerClient

    def client(handler):
        g = GlimmerClient(base_url="http://local.test/v1", api_key="k")
        g._http = httpx.Client(transport=httpx.MockTransport(handler))
        return g

    def refused(req):
        raise httpx.ConnectError("refused")
    ok, why = client(refused).health_detail()
    assert not ok and "start_glimmer" in why
    ok, why = client(lambda r: httpx.Response(503, json={})).health_detail()
    assert not ok and "loading" in why
    ok, why = client(lambda r: httpx.Response(200 if r.url.path == "/health" else 401, json={})).health_detail()
    assert not ok and "GLIMMER_API_KEY" in why
    ok, why = client(lambda r: httpx.Response(200, json={"data": []})).health_detail()
    assert ok


@pytest.mark.parametrize("answer,expect", [
    ("The moth is 3 to 4 cm wide [S1].", False),
    ("Hang the trap 1.5 m above the ground and use one trap for every 0.5 to 2 ha [S1].", False),
    ("Mix 10 ml per litre of water before use [S1].", True),
    ("Use 6 to 8 kg of neem kernels per hectare [S1].", True),
    ("Spray only when needed [S1].", True),
])
def test_pesticide_reminder_only_for_chemical_or_dose_answers(answer, expect):
    from fieldwise.retrieval import Passage
    p = [Passage("c", "d", "t", 1, "The moth is 3 to 4 cm wide. Hang the trap 1.5 m up, one trap per 0.5 to 2 ha. "
                                    "Mix 10 ml per litre. Use 6 to 8 kg of neem kernels per hectare. Spray only when needed.")]
    assert (PPE_REMINDER in check(answer, p).text) is expect
