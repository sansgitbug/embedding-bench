"""Retrieval metrics over ranked chunk IDs.

Ported from FinSight's src/evaluation/retrieval_metrics.py with the same
definitions; tests/test_finsight_parity.py checks the two agree.
"""

from __future__ import annotations

from typing import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the labelled relevant chunks that appear in the top k."""
    _check(relevant, k)
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1/rank of the first relevant chunk in the top k, else 0."""
    _check(relevant, k)
    for rank, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def is_correct(ranked: Sequence[str], relevant: set[str], k: int) -> bool:
    """A query is correct if any labelled relevant chunk is in the top k."""
    _check(relevant, k)
    return bool(set(ranked[:k]) & relevant)


def _check(relevant: set[str], k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not relevant:
        raise ValueError("relevant set cannot be empty")
