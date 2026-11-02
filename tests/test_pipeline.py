"""End-to-end pipeline tests with mocked Glimmer and Spark (no network)."""
import io
import json

import httpx
import pytest
from PIL import Image

from fieldwise import connectivity
from fieldwise.cache import DiskCache
from fieldwise.pipeline import Fieldwise
from fieldwise.providers.glimmer import GlimmerClient
from fieldwise.providers.spark import SparkClient
from fieldwise.queue import EscalationQueue
from fieldwise.retrieval import GuideIndex
from fieldwise.config import settings

from helpers import FAKE_CHUNKS, chat_response, mock_transport


class Recorder:
    def __init__(self):
        self.bodies = []


def make_app(tmp_path, glimmer_handler, spark_handler=None, spark_rec=None):
    g = GlimmerClient(base_url="http://local.test/v1", api_key="k")
    g._http = httpx.Client(transport=mock_transport(glimmer_handler))
    s = SparkClient()
    s.cache = DiskCache(tmp_path, "spark")

    def _spark(request, body):
        if spark_rec is not None:
            spark_rec.bodies.append(body)
        return spark_handler(request, body)

    if spark_handler:
        s._http = httpx.Client(transport=mock_transport(_spark))
    q = EscalationQueue(tmp_path / "q.sqlite3")
    return Fieldwise(index=GuideIndex(FAKE_CHUNKS), glimmer=g, spark=s, queue=q)


def spark_text(text):
    return lambda req, body: {
        "model": "muse-spark-1.3", "output_text": text,
        "usage": {"input_tokens": 1000, "output_tokens": 500, "input_tokens_details": {"cached_tokens": 0}},
    }


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    object.__setattr__(settings, "meta_api_key", "test-key")
    yield
    object.__setattr__(settings, "meta_api_key", None)


def test_confident_local_answer_stays_on_device(tmp_path):
    connectivity.set_forced_offline(True)
    app = make_app(tmp_path, lambda r, b: chat_response(
        "ANSWER: Scout the field twice a week and check 20 plants in a W pattern [S1].\nSUPPORTED: YES",
        {"YES": -0.02, "NO": -4.0}, marker="SUPPORTED:"))
    res = app.ask("How often should I scout maize for fall armyworm?", crop="maize")
    assert res.route == "local"
    assert res.confidence > 0.7
    assert res.sources and res.sources[0]["page"] == 3
    assert "Glimmer" in res.answered_by


def test_low_confidence_offline_is_queued_then_flushed(tmp_path):
    connectivity.set_forced_offline(True)
    rec = Recorder()
    app = make_app(
        tmp_path,
        lambda r, b: chat_response("ANSWER: Possibly scout the field [S1].\nSUPPORTED: YES",
                                   {"YES": -1.2, "NO": -0.4}, marker="SUPPORTED:"),
        spark_text("ANSWER: Scout the field twice a week [S1].\nSUPPORTED: YES"), rec)
    q = "My name is Kofi Mensah, call 0241234567. How often should I scout for fall armyworm?"
    res = app.ask(q, crop="maize")
    assert res.route == "queued" and res.queue_id
    assert res.answer.startswith("Tentative")
    stored = app.queue.pending()[0]["payload"]["question"]
    assert "Kofi" not in stored and "0241234567" not in stored

    connectivity.set_forced_offline(False)
    connectivity._state.update(checked=1e18, online=True)  # pretend the probe succeeded
    done = app.flush_queue()
    assert done and "error" not in done[0]
    sent = json.dumps(rec.bodies)
    assert "Kofi" not in sent and "0241234567" not in sent
    assert app.queue.pending() == []


def test_low_confidence_online_escalates_with_redaction(tmp_path):
    connectivity.set_forced_offline(False)
    connectivity._state.update(checked=1e18, online=True)
    rec = Recorder()
    app = make_app(
        tmp_path,
        lambda r, b: chat_response("ANSWER: Not sure.\nSUPPORTED: NO", {"YES": -0.9, "NO": -0.6}, marker="SUPPORTED:"),
        spark_text("ANSWER: Apply Product-A at 10 ml per 15 litres into the whorl [S2]. Also use 99 ml.\nSUPPORTED: YES"),
        rec)
    res = app.ask("I farm near Ejura. What dose of Product-A for fall armyworm in the maize whorl?", crop="maize")
    assert res.route == "cloud" and "Spark" in res.answered_by
    assert "Ejura" not in json.dumps(rec.bodies)
    assert "99 ml" not in res.answer                      # invented quantity removed
    assert "read and follow the product label" in res.answer
    assert res.cost_usd == pytest.approx((1000 * 1.25 + 500 * 4.25) / 1e6)


def test_photo_diagnosis_local_with_letter_probabilities(tmp_path):
    connectivity.set_forced_offline(True)
    app = make_app(tmp_path, lambda r, b: chat_response("ANSWER: B", {"B": -0.05, "F": -3.2, "Z": -4.0}))
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (20, 140, 40)).save(buf, format="JPEG")
    res = app.diagnose(buf.getvalue(), crop="maize")
    assert res.label == "maize/fall_armyworm"
    assert res.route == "local" and res.confidence > 0.9
    assert set(res.probabilities) >= {"maize/fall_armyworm"}


def test_unsure_choice_forces_escalation(tmp_path):
    connectivity.set_forced_offline(True)
    app = make_app(tmp_path, lambda r, b: chat_response("ANSWER: Z", {"Z": -0.1, "B": -2.5}))
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 200, 40)).save(buf, format="JPEG")
    res = app.diagnose(buf.getvalue(), crop="maize")
    assert res.confidence == 0.0 and res.route == "queued"


def _glimmer_down_app(tmp_path, spark_handler=None, rec=None):
    def boom(request):
        raise httpx.ConnectError("refused")
    app = make_app(tmp_path, lambda r, b: {}, spark_handler, rec)
    app.glimmer._http = httpx.Client(transport=httpx.MockTransport(boom))
    return app


def test_glimmer_down_offline_queues_and_never_claims_a_local_answer(tmp_path):
    connectivity.set_forced_offline(True)
    app = _glimmer_down_app(tmp_path)
    res = app.ask("My name is Kofi Mensah. How do I scout for fall armyworm?", crop="maize")
    assert res.route == "queued" and res.queue_id and res.confidence is None
    assert res.local_available is False and res.answered_by == "none"
    assert "isn't running" in res.note and "extension officer" in res.answer
    assert "Kofi" not in app.queue.pending()[0]["payload"]["question"]


def test_glimmer_down_online_goes_straight_to_spark_redacted(tmp_path):
    connectivity.set_forced_offline(False)
    connectivity._state.update(checked=1e18, online=True)
    rec = Recorder()
    app = _glimmer_down_app(tmp_path, spark_text("ANSWER: Scout the field twice a week [S1].\nSUPPORTED: YES"), rec)
    res = app.ask("I'm Kofi Mensah. How often should I scout for fall armyworm?", crop="maize")
    assert res.route == "cloud" and "Spark" in res.answered_by and res.confidence is None
    assert "Kofi" not in json.dumps(rec.bodies)
    assert "straight to Muse Spark" in res.note


def test_no_model_reachable_says_so_and_queues(tmp_path):
    connectivity.set_forced_offline(False)
    connectivity._state.update(checked=1e18, online=True)
    object.__setattr__(settings, "meta_api_key", None)      # Spark can't be called
    app = _glimmer_down_app(tmp_path)
    res = app.ask("How often should I scout for fall armyworm?", crop="maize")
    assert res.route == "unavailable" and res.answered_by == "none" and res.confidence is None
    assert "MODEL_API_KEY" in res.cloud["error"]
    assert res.queue_id and len(app.queue.pending()) == 1
    assert "Glimmer" in res.note and "Spark" in res.note


def test_low_confidence_with_failed_cloud_call_is_really_queued(tmp_path):
    connectivity.set_forced_offline(False)
    connectivity._state.update(checked=1e18, online=True)
    object.__setattr__(settings, "meta_api_key", None)
    app = make_app(tmp_path, lambda r, b: chat_response("ANSWER: Not sure.\nSUPPORTED: NO",
                                                         {"YES": -0.9, "NO": -0.6}, marker="SUPPORTED:"))
    res = app.ask("How often should I scout for fall armyworm?", crop="maize")
    assert res.route == "local_fallback" and res.answer.startswith("Tentative")
    assert res.queue_id and len(app.queue.pending()) == 1

def test_glimmer_timeout_is_reported_as_slow_not_as_off(tmp_path):
    connectivity.set_forced_offline(True)
    app = make_app(tmp_path, lambda r, b: {})

    def slow(request):
        raise httpx.ReadTimeout("timed out")
    app.glimmer._http = httpx.Client(transport=httpx.MockTransport(slow))
    res = app.ask("How often should I scout for fall armyworm?", crop="maize")
    assert "didn't answer within" in res.note and "isn't running" not in res.note
    assert "timed out" in res.local["error"]


def _blocked_spark(app):
    body = {"error": {"code": "content_policy_violation", "message": "Request blocked."}}
    app.spark._http = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(400, json=body)))


def test_spark_content_filter_is_not_queued_or_retried(tmp_path):
    connectivity.set_forced_offline(False)
    connectivity._state.update(checked=1e18, online=True)
    app = make_app(tmp_path, lambda r, b: chat_response("ANSWER: Not sure.\nSUPPORTED: NO",
                                                         {"YES": -0.9, "NO": -0.6}, marker="SUPPORTED:"),
                   spark_text("unused"))
    _blocked_spark(app)
    res = app.ask("Does fall armyworm only attack maize, or other crops too?", crop="maize")
    assert res.route == "local_fallback" and res.cloud["blocked"]
    assert res.queue_id is None and app.queue.pending() == []
    assert "content filter" in res.note and "queued" not in res.answer


def test_spark_content_filter_with_glimmer_down_gives_referral(tmp_path):
    connectivity.set_forced_offline(False)
    connectivity._state.update(checked=1e18, online=True)
    app = _glimmer_down_app(tmp_path, spark_text("unused"))
    _blocked_spark(app)
    res = app.ask("Is maize damaged by fall armyworm safe to eat?", crop="maize")
    assert res.route == "unavailable" and res.cloud["blocked"] and res.queue_id is None
    assert "extension officer" in res.answer and "content filter" in res.note


def test_restricted_key_is_reported_as_account_blocked(tmp_path):
    from fieldwise.providers.spark import SparkAccountBlocked
    connectivity.set_forced_offline(False)
    connectivity._state.update(checked=1e18, online=True)
    app = make_app(tmp_path, lambda r, b: {}, spark_text("unused"))
    body = {"error": {"code": "user_blocked", "message": "Your access has been restricted.", "type": "permission_error"}}
    app.spark._http = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(403, json=body)))
    with pytest.raises(SparkAccountBlocked):
        app.spark.chat("system", "question")


def test_spark_switched_off_is_not_called_unreachable(tmp_path):
    """Cache-only mode with nothing cached (e.g. the Kaggle hub, no key): say 'switched off', still queue."""
    connectivity.set_forced_offline(False)
    connectivity._state.update(checked=1e18, online=True)
    object.__setattr__(settings, "cache_only", True)
    try:
        app = make_app(tmp_path, lambda r, b: chat_response("ANSWER: Not sure.\nSUPPORTED: NO",
                                                            {"YES": -0.9, "NO": -0.6}, marker="SUPPORTED:"))
        res = app.ask("What should I do about the worms in my maize?", crop="maize")
        assert res.route == "local_fallback" and res.queue_id
        assert "switched off" in res.note and "couldn't be reached" not in res.note
        assert res.cloud["disabled"] and not res.cloud["blocked"]
    finally:
        object.__setattr__(settings, "cache_only", False)


def test_spark_without_key_is_switched_off_too(tmp_path):
    connectivity.set_forced_offline(False)
    connectivity._state.update(checked=1e18, online=True)
    object.__setattr__(settings, "meta_api_key", None)
    app = make_app(tmp_path, lambda r, b: chat_response("ANSWER: Not sure.\nSUPPORTED: NO",
                                                        {"YES": -0.9, "NO": -0.6}, marker="SUPPORTED:"))
    res = app.ask("What should I do about the worms in my maize?", crop="maize")
    assert "switched off" in res.note and res.route == "local_fallback"


def test_account_guard_pauses_spark_after_refusals(tmp_path):
    from fieldwise.providers.spark import SparkBlocked, SparkDisabled
    calls = []
    s = SparkClient()
    s.cache = DiskCache(tmp_path, "spark")
    s._http = httpx.Client(transport=httpx.MockTransport(
        lambda r: (calls.append(1), httpx.Response(400, json={"error": {"code": "content_policy_violation"}}))[1]))
    object.__setattr__(settings, "spark_max_refusals", 2)
    for i in range(2):
        with pytest.raises(SparkBlocked):
            s.chat("sys", f"question {i}")
    with pytest.raises(SparkDisabled, match="Paused after 2"):
        s.chat("sys", "question 3")
    assert len(calls) == 2                       # the third request never left the hub


def test_refused_key_stops_spark_and_sam(tmp_path):
    import io

    from PIL import Image

    from fieldwise.providers.sam import SamClient, SamUnavailable
    from fieldwise.providers.spark import SparkAccountBlocked
    calls = []
    s = SparkClient()
    s.cache = DiskCache(tmp_path, "spark")
    s._http = httpx.Client(transport=httpx.MockTransport(
        lambda r: (calls.append(1), httpx.Response(403, json={"error": {"code": "user_blocked"}}))[1]))
    with pytest.raises(SparkAccountBlocked):
        s.chat("sys", "q1")
    with pytest.raises(SparkAccountBlocked, match="earlier"):
        s.chat("sys", "q2")
    sam = SamClient()
    sam.cache = DiskCache(tmp_path, "sam")
    sam._http = httpx.Client(transport=httpx.MockTransport(lambda r: (calls.append(1), httpx.Response(200))[1]))
    buf = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buf, "JPEG")
    with pytest.raises(SamUnavailable, match="403"):
        sam.segment(buf.getvalue(), "leaf")
    assert len(calls) == 1                       # only the first Spark request was sent
