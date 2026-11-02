"""Tiny content-addressed disk cache for cloud calls.

Every Spark and SAM response is stored by a hash of its request. Benchmark
re-runs cost nothing, and the demo can replay a cached answer if the network
drops in front of the judges (FIELDWISE_CACHE_ONLY=1).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def request_key(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class DiskCache:
    def __init__(self, root: Path, namespace: str):
        self.dir = Path(root) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, value: Any) -> None:
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), "utf-8")
        tmp.replace(path)
