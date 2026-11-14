"""Client-side rate limits and persistent usage budgets.

Rate limits (per minute) make the harness wait. Budgets (calls per month,
total tokens) make it stop: they guard free quotas that do not refill within
a run, so the run ends with an error instead of spending into them.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from bench.retry import QuotaExhausted


class Ledger:
    """Every API attempt and every billed token this harness has sent for one provider.

    Persisted to disk so it survives across runs. It cannot see use of the
    same key outside this harness.
    """

    def __init__(self, path: Path, month=None):
        self.path = path
        self._month = month or (lambda: datetime.now(timezone.utc).strftime("%Y-%m"))
        if path.exists():
            self.state = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.state = {"calls_total": 0, "calls_by_month": {}, "tokens_total": 0}

    def calls_this_month(self) -> int:
        return self.state["calls_by_month"].get(self._month(), 0)

    @property
    def calls_total(self) -> int:
        return self.state["calls_total"]

    @property
    def tokens_total(self) -> int:
        return self.state["tokens_total"]

    def add_call(self, tokens_estimate: int = 0, inputs: int = 1, now: float | None = None) -> None:
        month = self._month()
        self.state["calls_by_month"][month] = self.state["calls_by_month"].get(month, 0) + 1
        self.state["calls_total"] += 1
        # Last minute of calls, so a new process starts with the provider's real window.
        now = time.time() if now is None else now
        recent = [e for e in self.state.get("recent", []) if now - e[0] < 60]
        self.state["recent"] = recent + [[now, tokens_estimate, inputs]]
        self._save()

    def add_tokens(self, tokens: int) -> None:
        self.state["tokens_total"] += tokens
        if self.state.get("recent"):
            self.state["recent"][-1][1] = tokens
        self._save()

    def recent(self, now: float | None = None) -> list[list]:
        now = time.time() if now is None else now
        return [e for e in self.state.get("recent", []) if now - e[0] < 60]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


class RateLimiter:
    def __init__(
        self,
        *,
        rpm: int | None = None,
        tpm: int | None = None,
        ipm: int | None = None,
        ledger: Ledger | None = None,
        max_calls_per_month: int | None = None,
        token_budget: int | None = None,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        if (max_calls_per_month is not None or token_budget is not None) and ledger is None:
            raise ValueError("budgets need a ledger")
        self.rpm, self.tpm, self.ipm = rpm, tpm, ipm
        self.ledger = ledger
        self.max_calls_per_month = max_calls_per_month
        self.token_budget = token_budget
        self._clock, self._sleep = clock, sleep
        self._events: deque[list] = deque()  # [time, tokens, inputs]
        if ledger is not None:
            # Seed the window with calls made by earlier processes in the last minute.
            wall, mono = time.time(), self._clock()
            for at, tokens, inputs in ledger.recent(wall):
                self._events.append([mono - (wall - at), tokens, inputs])

    def check_budget(self, calls: int, tokens: int) -> None:
        """Raise QuotaExhausted if `calls` more calls or `tokens` more tokens would break a budget."""
        if self.max_calls_per_month is not None:
            used = self.ledger.calls_this_month()
            if used + calls > self.max_calls_per_month:
                raise QuotaExhausted(
                    f"needs ~{calls} more API calls but {used} of the {self.max_calls_per_month} "
                    "calls/month budget are already used by this harness"
                )
        if self.token_budget is not None:
            used = self.ledger.tokens_total
            if used + tokens > self.token_budget:
                raise QuotaExhausted(
                    f"needs ~{tokens:,} more tokens but {used:,} of the {self.token_budget:,} "
                    "token budget are already used by this harness"
                )

    def acquire(self, tokens: int, inputs: int = 1) -> float:
        """Block until one request of ~`tokens` tokens and `inputs` texts fits.

        Records the attempt in the ledger before it is sent, so failed
        attempts count too. Returns the seconds spent waiting.
        """
        self.check_budget(1, tokens)
        for limit, need, unit in ((self.tpm, tokens, "tokens"), (self.ipm, inputs, "inputs")):
            if limit is not None and need > limit:
                raise ValueError(f"one request of {need} {unit} exceeds the {limit} {unit}/min cap")
        waited = 0.0
        while True:
            now = self._clock()
            while self._events and now - self._events[0][0] >= 60:
                self._events.popleft()
            over = (
                (self.rpm is not None and len(self._events) + 1 > self.rpm)
                or (self.tpm is not None and sum(e[1] for e in self._events) + tokens > self.tpm)
                or (self.ipm is not None and sum(e[2] for e in self._events) + inputs > self.ipm)
            )
            if not over:
                break
            pause = 60 - (now - self._events[0][0]) + 0.05
            self._sleep(pause)
            waited += pause
        self._events.append([self._clock(), tokens, inputs])
        if self.ledger is not None:
            self.ledger.add_call(tokens, inputs)
        return waited

    def record_tokens(self, tokens: int) -> None:
        """Replace the last request's estimate with the provider's real count."""
        if self._events:
            self._events[-1][1] = tokens
        if self.ledger is not None:
            self.ledger.add_tokens(tokens)
