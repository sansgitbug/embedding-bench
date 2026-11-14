"""Bounded retry with exponential backoff.

Only rate-limit and transient server/network errors are retried, a small fixed
number of times. Exhausting the attempts raises; nothing is ever dropped.
Quota exhaustion is never retried.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


class RateLimitError(Exception):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(Exception):
    """Server error or network failure worth retrying."""


class QuotaExhausted(RuntimeError):
    """A free quota or budget is used up (or would be). Stop; do not retry."""


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> tuple[T, int]:
    """Call fn until it succeeds. Returns (result, number of retries).

    Any exception other than RateLimitError / TransientError propagates
    immediately, including QuotaExhausted.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(), attempt - 1
        except (RateLimitError, TransientError) as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"gave up after {max_attempts} attempts (no more retries, to protect "
                    f"the quota): {exc}"
                ) from exc
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay *= 0.5 + random.random() / 2  # jitter
            retry_after = getattr(exc, "retry_after", None)
            if retry_after:
                delay = max(delay, min(retry_after, max_delay))
            log(f"    {type(exc).__name__}: {exc}; retry {attempt}/{max_attempts - 1} in {delay:.1f}s")
            sleep(delay)
    raise AssertionError("unreachable")
