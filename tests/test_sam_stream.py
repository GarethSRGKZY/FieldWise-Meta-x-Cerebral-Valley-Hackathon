"""Reading SAM 3.1's Responses stream. The event sequence below is copied from a live sam-3.1 reply
(30 Oct 2026): no text deltas, the result only in the done events. Text from Meta's protocol doc."""
import io
import json

import httpx
import pytest
from PIL import Image

from fieldwise.cache import DiskCache
from fieldwise.config import settings
from fieldwise.providers.sam import SamClient, SamUnavailable, collect_stream_text, rasterize

from helpers import sam_line

ITEM = "msg_2d01b5fd-3dfa-4727-8573-014c41395ed3"


def live_events(text: str, with_output_text_done: bool = True, deltas: list[str] | None = None) -> list[str]:
    """The live API's event sequence, with `text` as the output."""
    part = lambda t: {"annotations": [], "logprobs": [], "text": t, "type": "output_text"}  # noqa: E731
    ev = [
        {"type": "response.created", "sequence_number": 0, "response": {"status": "in_progress", "output": []}},
        {"type": "response.in_progress", "sequence_number": 1, "response": {"status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "output_index": 0, "sequence_number": 2,
         "item": {"content": [part("")], "id": ITEM, "role": "assistant", "status": "in_progress", "type": "message"}},
        {"type": "response.content_part.added", "content_index": 0, "item_id": ITEM, "output_index": 0,
         "part": part(""), "sequence_number": 3},
    ]
    for d in deltas or []:
        ev.append({"type": "response.output_text.delta", "content_index": 0, "item_id": ITEM, "output_index": 0, "delta": d})
    if with_output_text_done:
        ev.append({"type": "response.output_text.done", "content_index": 0, "item_id": ITEM, "output_index": 0,
                   "logprobs": [], "text": text, "sequence_number": 4})
    ev += [
        {"type": "response.content_part.done", "content_index": 0, "item_id": ITEM, "output_index": 0,
         "part": part(text), "sequence_number": 5},
        {"type": "response.output_item.done", "output_index": 0, "sequence_number": 6,
         "item": {"content": [part(text)], "id": ITEM, "role": "assistant", "status": "completed", "type": "message"}},
        {"type": "response.completed", "sequence_number": 7, "response": {"status": "completed",
         "usage": {"images_processed": 1, "input_tokens": 0, "output_tokens": 0}}},
    ]
    lines = []
    for e in ev:
        lines += [f"event: {e['type']}", "data: " + json.dumps(e), ""]
    return lines + ["data: [DONE]"]


LINE = sam_line([(10, 10, 49, 49, [[1, 1], [1, 1]]), (60, 5, 79, 24, [[1, 0], [0, 1]])], 100, 80)


def test_live_empty_reply_is_a_valid_no_match():
    assert collect_stream_text(live_events("")) == ""


@pytest.mark.parametrize("both_done_events", [True, False])
def test_result_only_in_done_events_is_read(both_done_events):
    raw = collect_stream_text(live_events(LINE, with_output_text_done=both_done_events))
    assert raw == LINE
    _, n, conf = rasterize(raw, 100, 80)
    assert n == 2 and conf == [0.9, 0.9]


def test_deltas_split_inside_a_mask_are_joined_exactly():
    cut = LINE.index("data=") + 9                       # inside the first mask payload
    raw = collect_stream_text(live_events(LINE, deltas=[LINE[:cut], LINE[cut:]]))
    assert raw == LINE and rasterize(raw, 100, 80)[1] == 2


def test_refusal_is_an_error_not_an_empty_result():
    lines = [ln.replace('"type": "output_text"', '"type": "refusal"') if "content_part.done" in ln else ln
             for ln in live_events("")]
    with pytest.raises(SamUnavailable, match="refused"):
        collect_stream_text(lines)


def test_client_reads_live_shaped_stream_and_caches_it(tmp_path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content="\n".join(live_events(LINE)).encode())

    object.__setattr__(settings, "meta_api_key", "test-key")
    try:
        sam = SamClient()
        sam.cache = DiskCache(tmp_path, "sam")
        sam._http = httpx.Client(transport=httpx.MockTransport(handler))
        buf = io.BytesIO()
        Image.new("RGB", (100, 80), (30, 130, 30)).save(buf, "JPEG")
        masks, cost, cached = sam.segment(buf.getvalue(), "leaf")
        assert masks.instances == 2 and not cached and cost == settings.sam_usd_per_image
        again, cost2, cached2 = sam.segment(buf.getvalue(), "leaf")
        assert cached2 and again.instances == 2 and cost2 == 0.0 and len(calls) == 1
        assert "cache_version" not in json.dumps(calls[0])          # the cache tag is never sent to Meta
    finally:
        object.__setattr__(settings, "meta_api_key", None)


def _client(tmp_path, handler):
    sam = SamClient()
    sam.cache = DiskCache(tmp_path, "sam")
    sam._http = httpx.Client(transport=httpx.MockTransport(handler))
    sam.RETRY_DELAYS_S = (0.0, 0.0)
    return sam


def _jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (100, 80), (30, 130, 30)).save(buf, "JPEG")
    return buf.getvalue()


def test_network_hiccup_is_retried(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("WinError 10060", request=request)
        return httpx.Response(200, content="\n".join(live_events(LINE)).encode())

    object.__setattr__(settings, "meta_api_key", "test-key")
    try:
        masks, _, _ = _client(tmp_path, handler).segment(_jpeg(), "leaf")
        assert masks.instances == 2 and len(calls) == 2
    finally:
        object.__setattr__(settings, "meta_api_key", None)


def test_blocked_key_is_not_retried(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(403, json={"error": {"code": "user_blocked"}})

    object.__setattr__(settings, "meta_api_key", "test-key")
    try:
        with pytest.raises(SamUnavailable, match="403"):
            _client(tmp_path, handler).segment(_jpeg(), "leaf")
        assert len(calls) == 1
    finally:
        object.__setattr__(settings, "meta_api_key", None)
