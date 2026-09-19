from __future__ import annotations

from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
import threading
import time
from typing import Any


def retrieval_fingerprint(
    *, thread_id: str, query: str, sources: list[str], filters: dict[str, Any], index_revision: int
) -> str:
    normalized_query = re.sub(r"\s+", " ", query).strip().casefold()
    payload = {
        "thread_id": thread_id,
        "query": normalized_query,
        "sources": sorted(sources),
        "filters": filters,
        "index_revision": index_revision,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class _Entry:
    expires_at: float
    value: dict[str, Any]


class RetrievalCache:
    def __init__(self, ttl_seconds: int = 7200, max_per_thread: int = 10):
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_per_thread = max(1, int(max_per_thread))
        self._values: dict[str, OrderedDict[str, _Entry]] = defaultdict(OrderedDict)
        self._revision = 0
        self._lock = threading.Lock()

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def invalidate(self) -> int:
        with self._lock:
            self._revision += 1
            self._values.clear()
            return self._revision

    def get(self, thread_id: str, key: str) -> dict[str, Any] | None:
        now = time.monotonic()
        with self._lock:
            bucket = self._values.get(thread_id)
            if not bucket:
                return None
            entry = bucket.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                bucket.pop(key, None)
                return None
            bucket.move_to_end(key)
            value = deepcopy(entry.value)
            value["cache_hit"] = True
            return value

    def put(self, thread_id: str, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            bucket = self._values[thread_id]
            bucket[key] = _Entry(time.monotonic() + self.ttl_seconds, deepcopy(value))
            bucket.move_to_end(key)
            while len(bucket) > self.max_per_thread:
                bucket.popitem(last=False)


_default_cache: RetrievalCache | None = None
_default_lock = threading.Lock()


def get_retrieval_cache(ttl_seconds: int = 7200) -> RetrievalCache:
    global _default_cache
    with _default_lock:
        if _default_cache is None:
            _default_cache = RetrievalCache(ttl_seconds=ttl_seconds)
        return _default_cache
