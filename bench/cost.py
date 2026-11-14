"""List-price cost arithmetic. Never reflects free tiers or discounts."""

from __future__ import annotations


def cost_usd(tokens: int, price_per_1m: float) -> float:
    return tokens / 1_000_000 * price_per_1m


def cost_per_1000_correct(query_cost_usd: float, correct: int) -> float | None:
    """Query-embedding cost per 1,000 correct answers; None if nothing was correct."""
    if correct == 0:
        return None
    return query_cost_usd / correct * 1000
