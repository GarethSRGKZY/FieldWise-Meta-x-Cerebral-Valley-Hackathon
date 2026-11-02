"""A STUB that imitates llama.cpp's API so you can build the UI before the
17 GB Glimmer download finishes. It is not a model and its answers are canned.

    python scripts/fake_glimmer.py          # listens on 127.0.0.1:8080

The benchmark refuses to run against it (it reports model id FAKE-STUB).
"""
from __future__ import annotations

import re

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()
MODEL_ID = "FAKE-STUB-not-glimmer"


def tok(text, lp=-0.01, top=None):
    return {"token": text, "logprob": lp, "top_logprobs": top or []}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL_ID}]}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    user = body["messages"][-1]["content"]
    text = user if isinstance(user, str) else " ".join(p.get("text", "") for p in user if p.get("type") == "text")

    if "Options:" in text:
        # photo: pick B with high confidence unless the note says "unsure"
        note = re.search(r"Farmer's note: (.*)", text)
        unsure = bool(note and "unsure" in note.group(1).lower())
        letter, top = ("Z", [tok(" Z", -0.3), tok(" B", -1.5)]) if unsure else ("B", [tok(" B", -0.08), tok(" E", -2.9), tok(" Z", -4.2)])
        content = f"ANSWER: {letter}"
        lps = [tok("ANSWER"), tok(":"), tok(" " + letter, top[0]["logprob"], top)]
    else:
        m = re.search(r"\[S1\] \(.*?\)\n(.+?)(?:\n\n\[S2\]|\n\n(?:Crop|Question):)", text, re.S)
        first = re.split(r"(?<=[.!?])\s", m.group(1).strip())[0] if m else ""
        hard = "unsure" in text.split("Question:")[-1].lower() or not first
        if hard:
            content = "ANSWER: The passages may not cover this.\nSUPPORTED: NO"
            top = [tok(" NO", -0.7), tok(" YES", -0.8)]
            val = " NO"
        else:
            content = f"ANSWER: {first} [S1]\nSUPPORTED: YES"
            top = [tok(" YES", -0.05), tok(" NO", -3.2)]
            val = " YES"
        lps = [tok(content.rsplit("SUPPORTED:", 1)[0]), tok("SUPPORTED"), tok(":"), tok(val, top[0]["logprob"], top)]

    return {
        "model": MODEL_ID,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content, "reasoning_content": "(stub: no reasoning)"},
                     "logprobs": {"content": lps} if body.get("logprobs") else None}],
        "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": 20},
    }


@app.post("/v1/responses")
async def fake_spark(req: Request):
    """Spark-shaped stub so the cloud path can be smoke-tested without credits.
    Use with META_BASE_URL=http://127.0.0.1:8080/v1 MODEL_API_KEY=stub. Never for real numbers."""
    body = await req.json()
    text = " ".join(c.get("text", "") for m in body["input"] for c in m["content"] if c.get("type") == "input_text")
    if "Options:" in text:
        out = "ANSWER: B"
    else:
        m = re.search(r"\[S1\] \(.*?\)\n(.+?)(?:\n\n\[S2\]|\n\n(?:Crop|Question):)", text, re.S)
        first = re.split(r"(?<=[.!?])\s", m.group(1).strip())[0] if m else ""
        out = f"ANSWER: {first} [S1]\nSUPPORTED: YES" if first else "ANSWER: Not covered.\nSUPPORTED: NO"
    return {"model": "FAKE-STUB-spark", "output_text": out,
            "usage": {"input_tokens": len(text) // 4, "output_tokens": 300, "input_tokens_details": {"cached_tokens": 0}}}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
