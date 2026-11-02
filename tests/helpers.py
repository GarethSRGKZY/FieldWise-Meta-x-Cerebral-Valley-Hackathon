"""Test fixtures. Everything agronomic here is FAKE ("Product-A", made-up doses)
and exists only to exercise the code paths. Never ship it as guide content."""
from __future__ import annotations

import json

import httpx
from meta_sam_parser._mask_codec import _encode_raster  # test-only use of the private encoder

FAKE_CHUNKS = [
    {"chunk_id": "t1#0", "doc_id": "t1", "title": "TEST FIXTURE guide", "page": 3, "licence": "test",
     "text": "Fall armyworm larvae feed inside the maize whorl and leave ragged holes and sawdust-like frass. "
             "Scout the field twice a week and check 20 plants in a W pattern."},
    {"chunk_id": "t1#1", "doc_id": "t1", "title": "TEST FIXTURE guide", "page": 4, "licence": "test",
     "text": "TEST FIXTURE ONLY: Product-A may be applied at 10 ml per 15 litres of water into the whorl. "
             "Follow the label and wear protective equipment."},
    {"chunk_id": "t1#2", "doc_id": "t1", "title": "TEST FIXTURE guide", "page": 9, "licence": "test",
     "text": "Cassava mosaic disease causes yellow and green mosaic patterns and distorted leaves. "
             "Use clean planting material and remove infected plants."},
]


def chat_response(content: str, letter_logprobs: dict[str, float] | None = None, marker: str = "ANSWER:",
                  reasoning: str = "thinking...") -> dict:
    """Mimic llama.cpp's OpenAI-compatible chat response, with optional logprobs."""
    logprobs = None
    if letter_logprobs is not None:
        tokens = []
        head, _, tail = content.rpartition(marker)
        for piece in [head, marker]:
            if piece:
                tokens.append({"token": piece, "logprob": -0.01, "top_logprobs": []})
        value = tail.strip()
        top = [{"token": " " + k, "logprob": v} for k, v in letter_logprobs.items()]
        tokens.append({"token": " " + value, "logprob": letter_logprobs.get(value, -5.0), "top_logprobs": top})
        logprobs = {"content": tokens}
    return {
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content, "reasoning_content": reasoning},
                     "logprobs": logprobs}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def mock_transport(handler) -> httpx.MockTransport:
    def _h(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        return httpx.Response(200, json=handler(request, body))
    return httpx.MockTransport(_h)


def sam_line(objects: list[tuple[int, int, int, int, list[list[int]]]], w: int, h: int, conf: float = 0.9) -> str:
    """Build a SAM 3.1 wire line. Each object: (x1, y1, x2, y2, mask_rows) with x2/y2 inclusive."""
    recs = []
    for i, (x1, y1, x2, y2, rows) in enumerate(objects):
        mh, mw = len(rows), len(rows[0])
        flat = [v for r in rows for v in r]
        payload = _encode_raster(flat, mw, mh)
        recs.append(f"{i}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h};c={conf}|>"
                    f"<|mask;x=0;y=0;c={conf};data={mh},{mw},{payload}|>")
    return "<0f>" + ",".join(recs)
