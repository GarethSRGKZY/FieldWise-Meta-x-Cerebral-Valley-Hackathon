"""Muse Glimmer, running locally behind llama.cpp's OpenAI-compatible server.

Start the server (from the Meta docs, llama.cpp b10353 or newer):

    llama serve -hf meta-models/Muse-Glimmer-30B-GGUF \
      --hf-file Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
      -a muse-glimmer -ngl 99 -c 32768 -np 1 \
      --host 127.0.0.1 --port 8080 --api-key "$GLIMMER_API_KEY" \
      --chat-template-kwargs '{"reasoning_strength":"low"}'

Docs-backed choices in this client:
- No temperature=0: the docs say greedy decoding loops on this model, so we
  keep the published sampling settings and fix a seed instead.
- reasoning_strength goes through chat_template_kwargs (llama.cpp ignores
  OpenAI's reasoning_effort).
- Reasoning arrives in message.reasoning_content; content stays clean.

Not in Meta's docs (llama.cpp server behaviour, verify on your build):
- `logprobs` / `top_logprobs` on /v1/chat/completions
- `seed`
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

import httpx

from ..config import settings


class GlimmerUnavailable(RuntimeError):
    """Glimmer gave no usable answer. `.short` is a plain sentence for the farmer-facing note;
    str(e) keeps the technical detail for the hub operator."""

    def __init__(self, detail: str, short: str = "The hub model (Muse Glimmer) isn't running."):
        super().__init__(detail)
        self.short = short


@dataclass
class LocalResult:
    content: str
    reasoning: str
    logprobs: list[dict] | None
    finish_reason: str | None
    latency_s: float
    usage: dict = field(default_factory=dict)


def image_data_url(jpeg_bytes: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(jpeg_bytes).decode('ascii')}"


class GlimmerClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.base_url = (base_url or settings.glimmer_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.glimmer_api_key
        self.model = model or settings.glimmer_model
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # trust_env=False: never route local traffic through a proxy.
        self._http = httpx.Client(timeout=settings.glimmer_timeout_s, headers=headers, trust_env=False)

    def health(self) -> bool:
        return self.health_detail()[0]

    def health_detail(self) -> tuple[bool, str]:
        """(ok, plain-language reason). Tells apart: nothing running, still loading, wrong key."""
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        try:
            r = self._http.get(f"{root}/health", timeout=2.0)
        except httpx.HTTPError:
            return False, (f"Nothing is answering at {root}. Start Glimmer with scripts/start_glimmer "
                           "in its own terminal and leave that window open.")
        if r.status_code == 503:
            return False, "Glimmer is still loading the model. Wait until its window says the server is listening."
        if r.status_code != 200:
            return False, f"The Glimmer server at {root} answered HTTP {r.status_code} on /health."
        try:
            m = self._http.get(f"{self.base_url}/models", timeout=2.0)
        except httpx.HTTPError:
            return True, "Running."
        if m.status_code in (401, 403):
            return False, ("Glimmer is running but rejected our key: GLIMMER_API_KEY in .env must match the one "
                           "the server was started with. Restart the start script after editing .env.")
        return True, "Running."

    def chat(
        self,
        system: str,
        user_text: str,
        image_jpeg: bytes | None = None,
        want_logprobs: bool = True,
        top_logprobs: int = 10,
        max_tokens: int | None = None,
    ) -> LocalResult:
        content: list[dict] | str
        if image_jpeg is not None:
            content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url(image_jpeg)}},
            ]
        else:
            content = user_text

        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens or settings.glimmer_max_tokens,
            "chat_template_kwargs": {"reasoning_strength": settings.glimmer_reasoning_strength},
            "seed": settings.glimmer_seed,
            "stream": False,
        }
        if want_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs

        t0 = time.perf_counter()
        try:
            r = self._http.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.TimeoutException as e:
            waited = time.perf_counter() - t0
            raise GlimmerUnavailable(f"Glimmer timed out after {waited:.0f} s (GLIMMER_TIMEOUT_S={settings.glimmer_timeout_s:.0f})",
                                     f"The hub model (Muse Glimmer) didn't answer within {waited:.0f} seconds.") from e
        except httpx.ConnectError as e:
            raise GlimmerUnavailable(f"Nothing listening at {self.base_url}: {e}") from e
        except httpx.HTTPError as e:
            raise GlimmerUnavailable(f"Glimmer connection failed at {self.base_url}: {e}",
                                     "The hub model (Muse Glimmer) couldn't be reached.") from e
        latency = time.perf_counter() - t0
        if r.status_code == 503:
            raise GlimmerUnavailable(f"Glimmer returned HTTP 503: {r.text[:300]}",
                                     "The hub model (Muse Glimmer) is still loading.")
        if r.status_code >= 400:
            raise GlimmerUnavailable(f"Glimmer returned HTTP {r.status_code}: {r.text[:300]}",
                                     f"The hub model (Muse Glimmer) returned an error (HTTP {r.status_code}).")

        data = r.json()
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        lp = (choice.get("logprobs") or {}).get("content") if choice.get("logprobs") else None
        return LocalResult(
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
            logprobs=lp,
            finish_reason=choice.get("finish_reason"),
            latency_s=latency,
            usage=data.get("usage") or {},
        )
