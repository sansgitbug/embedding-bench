"""The one interface every embedding provider implements."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import httpx
import numpy as np

from bench.retry import QuotaExhausted, RateLimitError, TransientError

Kind = Literal["document", "query"]

# Words that mark a 429 as "quota used up" rather than "slow down". Those are
# never retried: retrying cannot help and may burn more of the quota.
QUOTA_HINTS = ("per month", "/month", "monthly", "trial key", "balance", "insufficient", "quota")


@dataclass
class EmbedResult:
    vectors: np.ndarray  # shape (len(texts), dim)
    tokens: int  # tokens the provider says it processed (what it would bill)
    tokens_source: str  # where that number came from


class Provider(ABC):
    """Turn texts into vectors. One instance per benchmark run.

    Adapters make exactly one attempt per embed() call and signal retryable
    failures with RateLimitError / TransientError. The harness owns retrying,
    throttling, budgets, caching and timing, so those are identical for every
    provider.
    """

    #: Bump when the request format changes in a way that changes the vectors.
    cache_version = "1"

    def __init__(self, run_cfg: dict):
        self.cfg = run_cfg

    @abstractmethod
    def embed(self, texts: list[str], kind: Kind, model: str) -> EmbedResult: ...

    def describe(self) -> dict:
        """Extra facts for the run's metadata file (device, versions, ...)."""
        return {}


def require_key(name: str) -> str:
    key = os.environ.get(name, "").strip()
    if not key:
        raise SystemExit(f"{name} is not set. Copy .env.example to .env and add your key.")
    return key


def post_json(client: httpx.Client, url: str, *, headers: dict, body: dict) -> dict:
    """POST and map HTTP failures onto the harness's retry and stop signals."""
    try:
        response = client.post(url, headers=headers, json=body)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise TransientError(f"network error: {type(exc).__name__}") from exc
    status, text = response.status_code, response.text[:500]
    if status == 402 or (status == 429 and any(h in text.lower() for h in QUOTA_HINTS)):
        raise QuotaExhausted(f"HTTP {status} from {url}: {text}")
    if status == 429:
        retry_after = _seconds(response.headers.get("retry-after"))
        if retry_after is None and "per minute" in text.lower():
            retry_after = 61.0  # a per-minute window only clears after a minute
        raise RateLimitError(f"HTTP 429: {text[:300]}", retry_after=retry_after)
    if status >= 500:
        raise TransientError(f"HTTP {status}: {text[:300]}")
    if status != 200:
        raise RuntimeError(f"HTTP {status} from {url}: {text}")
    return response.json()


def _seconds(header: str | None) -> float | None:
    try:
        return float(header) if header else None
    except ValueError:
        return None
