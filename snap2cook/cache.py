"""Two-level cache: in-memory LRU (L1) + JSON file cache with TTL (L2)."""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from snap2cook.exceptions import FileOperationError
from snap2cook.fileio.safe_io import SafeFile

log = logging.getLogger("snap2cook.cache")


class Cache(ABC):
    """Key/value store of JSON-safe values. ``get`` returns None on a miss."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return the value or None."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value with an optional time-to-live in seconds."""


class LRUCache(Cache):
    """Least-recently-used in-memory cache built on OrderedDict."""

    def __init__(self, max_size: int = 256, clock: Callable[[], float] = time.time) -> None:
        self._max = max_size
        self._clock = clock
        self._data: OrderedDict[str, tuple[float | None, Any]] = OrderedDict()
        self._lock = threading.Lock()  # sessions run in different threads

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires, value = entry
            if expires is not None and expires < self._clock():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        with self._lock:
            self._data[key] = (self._clock() + ttl if ttl else None, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)


class FileCache(Cache):
    """JSON files on disk (via SafeFile). Any problem is treated as a miss."""

    def __init__(self, directory: Path, clock: Callable[[], float] = time.time) -> None:
        self._dir = directory
        self._clock = clock

    def _file(self, key: str) -> SafeFile:
        return SafeFile(self._dir / (hashlib.sha256(key.encode()).hexdigest() + ".json"))

    def get(self, key: str) -> Any | None:
        file = self._file(key)
        if not file.exists():
            return None
        try:
            entry = file.read_json()
        except FileOperationError as exc:
            log.warning("cache read failed: %s", exc.message)
            return None
        expires = entry.get("expires")
        if expires is not None and expires < self._clock():
            return None
        return entry.get("value")

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        try:
            self._file(key).write_json({"expires": self._clock() + ttl if ttl else None, "value": value})
        except (FileOperationError, TypeError) as exc:
            log.warning("cache write skipped: %s", exc)


class TieredCache(Cache):
    """L1 then L2; promotes L2 hits to L1 and tracks the hit rate."""

    def __init__(self, l1: Cache, l2: Cache | None = None) -> None:
        self._l1, self._l2 = l1, l2
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups answered from cache (0.0 when unused)."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def stats(self) -> tuple[int, int]:
        """(hits, misses)."""
        return self._hits, self._misses

    def get(self, key: str) -> Any | None:
        value = self._l1.get(key)
        if value is None and self._l2 is not None:
            value = self._l2.get(key)
            if value is not None:
                self._l1.set(key, value)
        if value is None:
            self._misses += 1
        else:
            self._hits += 1
        return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._l1.set(key, value, ttl)
        if self._l2 is not None:
            self._l2.set(key, value, ttl)
