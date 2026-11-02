"""Is the cloud reachable right now? A 1.5 s TCP probe, cached for 10 s.

FIELDWISE_FORCE_OFFLINE=1 (or the toggle in the UI) forces offline mode,
as a backup if real airplane mode misbehaves during the demo.
"""
from __future__ import annotations

import socket
import time
from urllib.parse import urlparse

from .config import settings

_state = {"checked": 0.0, "online": False, "forced": settings.force_offline}


def set_forced_offline(value: bool) -> None:
    _state["forced"] = value
    _state["checked"] = 0.0


def forced_offline() -> bool:
    return bool(_state["forced"])


def is_online(ttl_s: float = 10.0) -> bool:
    if _state["forced"]:
        return False
    now = time.monotonic()
    if now - _state["checked"] < ttl_s:
        return _state["online"]
    host = urlparse(settings.meta_base_url).hostname or "api.meta.ai"
    try:
        with socket.create_connection((host, 443), timeout=1.5):
            online = True
    except OSError:
        online = False
    _state.update(checked=now, online=online)
    return online
