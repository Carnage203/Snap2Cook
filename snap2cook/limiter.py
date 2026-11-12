"""Rate limiter protecting the free Gemini quota."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable

from snap2cook.exceptions import RateLimitExceededError


class RateLimiter:
    """Per-session sliding window plus a global daily ceiling."""

    def __init__(self, max_calls: int = 6, window_s: int = 600, daily_max: int = 300,
                 clock: Callable[[], float] = time.time) -> None:
        self._max, self._window, self._daily_max = max_calls, window_s, daily_max
        self._clock = clock
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._day = int(clock() // 86400)
        self._day_count = 0
        self._lock = threading.Lock()

    def check(self, session_id: str) -> None:
        """Record one generation or raise RateLimitExceededError."""
        with self._lock:
            now = self._clock()
            today = int(now // 86400)
            if today != self._day:
                self._day, self._day_count = today, 0
            if self._day_count >= self._daily_max:
                raise RateLimitExceededError("Global daily limit reached", retry_after=3600)
            window = self._calls[session_id]
            while window and window[0] <= now - self._window:
                window.popleft()
            if len(window) >= self._max:
                raise RateLimitExceededError(
                    f"Session limit of {self._max} per {self._window}s reached",
                    retry_after=int(window[0] + self._window - now) + 1)
            window.append(now)
            self._day_count += 1
