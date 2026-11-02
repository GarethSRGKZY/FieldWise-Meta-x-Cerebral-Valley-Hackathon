"""Replay mode: Glimmer's recorded benchmark outputs stand in for the live model in the demo video."""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import httpx
import pytest
from PIL import Image, ImageDraw

from fieldwise import connectivity
from fieldwise.cache import DiskCache
from fieldwise.config import settings
from fieldwise.pipeline import Fieldwise
from fieldwise.providers.glimmer import GlimmerClient
from fieldwise.providers.spark import SparkClient
from fieldwise.queue import EscalationQueue
from fieldwise.replay import GlimmerReplay
from fieldwise.retrieval import GuideIndex

from helpers import FAKE_CHUNKS, mock_transport

Q1 = "How often should I scout maize for fall armyworm?"
Q3 = "Which plants should I check for fall armyworm larvae in the whorl?"


def leaf_jpeg(seed: int, quality: int = 92, size=(640, 480)) -> bytes:
    im = Image.new("RGB", (640, 480), (40, 120 + seed * 20, 40))
    d = ImageDraw.Draw(im)
    for i in range(12):
        x = (i * 53 + seed * 97) % 600
        d.ellipse([x, (i * 37 + seed * 13) % 440, x + 40 + seed * 5, (i * 37 + seed * 13) % 440 + 30], fill=(120, 80 + i * 8, 20))
    if size != (640, 480):
        im = im.resize(size)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


@pytest.fixture
def replay(tmp_path):
    results, data, root = tmp_path / "results", tmp_path / "data", tmp_path / "CCMT"
    results.mkdir(), data.mkdir()
    qs = [{"id": "q1", "crop": "maize", "question": Q1, "answerable": True},
          {"id": "q2", "crop": "maize", "question": "What is FAW?", "answerable": True},
          {"id": "q3", "crop": "maize", "question": Q3, "answerable": True}]
    (data / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in qs))
    recs = [{"id": "q1", "confidence": 0.95, "supported": True, "cited": [1], "latency_s": 21.6, "ts": "2026-10-28T03:00:00+00:00",
             "confidence_source": "logprobs", "answer_shown": "Scout the field twice a week and check 20 plants in a W pattern [S1]."},
            {"id": "q2", "confidence": 0.95, "supported": True, "cited": [1], "answer_shown": "Fall armyworm [S1]."},
            {"id": "q3", "error": "timeout"},
            {"id": "q3", "confidence": 0.40, "supported": True, "cited": [1], "latency_s": 30.2,
             "answer_shown": "Check 20 plants in a W pattern [S1]."}]
    (results / "qa_local.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    img = root / "Raw Data" / "Maize" / "leaf spot" / "a.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(leaf_jpeg(1))
    with (data / "m.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "split", "crop", "label", "path", "sha1"])
        w.writeheader()
        w.writerow({"id": "i1", "split": "test", "crop": "maize", "label": "maize/leaf_spot",
                    "path": "Raw Data/Maize/leaf spot/a.jpg", "sha1": hashlib.sha1(img.read_bytes()).hexdigest()})
    (results / "images_local.jsonl").write_text(json.dumps(
        {"id": "i1", "pred": "maize/leaf_spot", "confidence": 0.9, "probabilities": {"maize/leaf_spot": 0.9},
         "confidence_source": "logprobs", "latency_s": 40.0}))
    return GlimmerReplay(results, data / "questions.jsonl", data / "m.csv", root, tmp_path / "h.json"), img


def test_question_matching(replay):
    rp, _ = replay
    assert rp.qa(Q1, "maize").id == "q1"
    assert rp.qa("  how often should i SCOUT maize for fall armyworm ", "maize").id == "q1"
    assert rp.qa(f"My name is Kofi Mensah, call me on 0244 123 456. {Q1}", "maize").id == "q1"   # redaction demo
    assert rp.qa(Q1, "cassava") is None                                    # different crop, different context
    assert rp.qa("Tell me what is FAW? please", "maize") is None           # short questions must match exactly
    assert rp.qa("What is FAW?", "maize").id == "q2"
    assert rp.qa(Q3, "maize").rec["confidence"] == 0.40                     # errored record skipped, good one used


def test_photo_matching(replay):
    rp, img = replay
    assert rp.dx(img.read_bytes(), "maize").id == "i1"
    recompressed = leaf_jpeg(1, quality=60, size=(320, 240))                # what a phone or chat app might do
    assert rp.dx(recompressed, "maize").id == "i1"
    assert rp.dx(leaf_jpeg(5), "maize") is None                             # a different photo
    assert rp.dx(img.read_bytes(), "cassava") is None


def make_app(tmp_path, rp):
    calls = []
    g = GlimmerClient(base_url="http://local.test/v1", api_key="k")
    g._http = httpx.Client(transport=mock_transport(lambda r, b: calls.append(b) or {}))
    s = SparkClient()
    s.cache = DiskCache(tmp_path, "spark")
    app = Fieldwise(index=GuideIndex(FAKE_CHUNKS), glimmer=g, spark=s, queue=EscalationQueue(tmp_path / "q.sqlite3"), replay=rp)
    return app, calls


def test_replayed_answer_is_labelled_and_never_calls_glimmer(tmp_path, replay):
    connectivity.set_forced_offline(True)
    app, calls = make_app(tmp_path, replay[0])
    res = app.ask(Q1, crop="maize")
    assert res.route == "local" and "replayed" in res.answered_by
    assert res.replayed["latency_s"] == 21.6 and res.replayed["benchmark_id"] == "q1"
    assert res.sources and res.sources[0]["page"] == 3
    assert calls == []
    assert app.glimmer_health()[0]


def test_replay_low_confidence_offline_queues_redacted_copy(tmp_path, replay):
    connectivity.set_forced_offline(True)
    app, _ = make_app(tmp_path, replay[0])
    res = app.ask(f"My name is Kofi Mensah, call me on 0244 123 456. {Q3}", crop="maize")
    assert res.route == "queued" and res.answer.startswith("Tentative") and res.replayed
    stored = app.queue.pending()[0]["payload"]["question"]
    assert "Kofi" not in stored and "0244" not in stored


def test_unrecorded_question_says_so_instead_of_inventing(tmp_path, replay):
    connectivity.set_forced_offline(True)
    app, calls = make_app(tmp_path, replay[0])
    res = app.ask("Why are my cassava leaves curling?", crop="maize")
    assert res.local_available is False and res.confidence is None and res.route == "queued"
    assert "Demo replay mode" in res.note and calls == []


def test_replayed_photo(tmp_path, replay):
    connectivity.set_forced_offline(True)
    app, _ = make_app(tmp_path, replay[0])
    res = app.diagnose(replay[1].read_bytes(), "maize")
    assert res.label == "maize/leaf_spot" and res.route == "local" and res.replayed["latency_s"] == 40.0


def test_benchmark_refuses_replay_mode(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    import run_bench
    object.__setattr__(settings, "glimmer_replay", True)
    monkeypatch.setattr(sys, "argv", ["run_bench.py", "--task", "qa", "--setup", "local"])
    try:
        with pytest.raises(SystemExit, match="REPLAY"):
            run_bench.main()
    finally:
        object.__setattr__(settings, "glimmer_replay", False)
