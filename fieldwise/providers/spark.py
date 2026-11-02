"""Muse Spark on the Meta Model API (Responses API), used only for escalations.

Per the docs: base URL https://api.meta.ai/v1, Bearer MODEL_API_KEY,
model `muse-spark-1.3` (Standard mode), images as `input_image` with a base64
data URL, answer in `output_text`, reasoning tokens billed as output.

Every response is cached on disk by request hash. With FIELDWISE_CACHE_ONLY=1
the client never touches the network, so a rehearsed demo can't break.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import httpx

from ..cache import DiskCache, request_key
from ..config import settings
from .glimmer import image_data_url


class SparkUnavailable(RuntimeError):
    pass


class SparkAccountBlocked(SparkUnavailable):
    """The Meta API refused the key itself (HTTP 401/403, e.g. 'user_blocked'). Nothing will work until
    that is resolved with Meta, so callers stop instead of sending more requests."""


class SparkDisabled(SparkUnavailable):
    """Cloud second opinions are switched off on this hub (cache-only mode with nothing cached, or no key).
    Not an outage, so the app must say "switched off", not "couldn't be reached"."""


class SparkBlocked(SparkUnavailable):
    """Spark's content filter refused this request (HTTP 400 content_policy). Retrying won't help,
    so callers must not queue or retry it; the farmer gets the extension-officer referral instead."""


@dataclass
class CloudResult:
    content: str
    latency_s: float
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    cached: bool = False
    model: str = ""


def spark_cost(usage: dict) -> float:
    inp = int(usage.get("input_tokens") or 0)
    cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    return (
        (inp - cached) * settings.spark_usd_per_m_input
        + cached * settings.spark_usd_per_m_cached
        + out * settings.spark_usd_per_m_output
    ) / 1_000_000


def _output_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts = []
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if c.get("type") == "output_text":
                parts.append(c.get("text") or "")
    return "".join(parts)


# Shared by every client in this process (Spark and SAM use the same key).
_GUARD = {"refusals": 0, "blocked": ""}


def note_key_refused(detail: str) -> None:
    """Meta refused the key (401/403): stop all further Meta API calls from this process."""
    _GUARD["blocked"] = detail[:300]


def key_refused() -> str:
    return _GUARD["blocked"]


def reset_guard() -> None:
    _GUARD.update(refusals=0, blocked="")


class SparkClient:
    def __init__(self):
        self.cache = DiskCache(settings.cache_dir, "spark")
        self._http = httpx.Client(timeout=settings.spark_timeout_s)

    def _body(self, system: str, user_text: str, image_jpeg: bytes | None) -> dict:
        # The docs only show user-role messages for the Responses API, so the
        # system text is prepended to the user turn rather than relying on an
        # undocumented `instructions` field.
        content: list[dict] = [{"type": "input_text", "text": f"{system}\n\n{user_text}"}]
        if image_jpeg is not None:
            content.append({"type": "input_image", "image_url": image_data_url(image_jpeg)})
        body: dict = {
            "model": settings.spark_model,
            "input": [{"type": "message", "role": "user", "content": content}],
        }
        if settings.spark_reasoning_effort:
            body["reasoning"] = {"effort": settings.spark_reasoning_effort}
        return body

    def chat(self, system: str, user_text: str, image_jpeg: bytes | None = None) -> CloudResult:
        body = self._body(system, user_text, image_jpeg)
        key = request_key(body)
        hit = self.cache.get(key)
        if hit is not None:
            return CloudResult(hit["content"], hit["latency_s"], hit["usage"], hit["cost_usd"], True, hit["model"])
        if settings.cache_only:
            raise SparkDisabled("Cache-only mode (FIELDWISE_CACHE_ONLY=1) and this request was never cached.")
        if not settings.meta_api_key:
            raise SparkDisabled("MODEL_API_KEY is not set.")
        if _GUARD["blocked"]:
            raise SparkAccountBlocked(f"Not sent: Meta refused the key earlier in this session ({_GUARD['blocked']}).")
        if settings.spark_max_refusals and _GUARD["refusals"] >= settings.spark_max_refusals:
            raise SparkDisabled(f"Paused after {_GUARD['refusals']} content-filter refusals to protect the account; "
                                "restart the app to resume.")

        headers = {"Authorization": f"Bearer {settings.meta_api_key}", "Content-Type": "application/json"}
        delay = 0.5
        last_err = ""
        for attempt in range(4):
            t0 = time.perf_counter()
            try:
                r = self._http.post(f"{settings.meta_base_url}/responses", json=body, headers=headers)
            except httpx.HTTPError as e:
                last_err = str(e)
            else:
                latency = time.perf_counter() - t0
                if r.status_code == 200:
                    data = r.json()
                    usage = data.get("usage") or {}
                    result = CloudResult(_output_text(data), latency, usage, spark_cost(usage), False, data.get("model", settings.spark_model))
                    self.cache.put(key, result.__dict__ | {"cached": False})
                    return result
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (401, 403):
                    note_key_refused(f"HTTP {r.status_code}: {r.text[:200]}")
                    raise SparkAccountBlocked(f"Meta API refused the key (HTTP {r.status_code}): {r.text[:300]}")
                if r.status_code == 400 and "content_polic" in r.text:
                    _GUARD["refusals"] += 1
                    raise SparkBlocked(f"Spark content filter declined the request: {r.text[:500]}")
                if r.status_code not in (429, 500, 502, 503, 504):
                    break  # 4xx other than 429 won't succeed on retry
                retry_after = r.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
            time.sleep(delay + random.uniform(0, delay / 2))
            delay *= 2
        raise SparkUnavailable(f"Spark call failed: {last_err}")
