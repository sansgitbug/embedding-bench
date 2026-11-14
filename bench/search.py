"""Exact cosine-similarity search. No ANN index, no reranking."""

from __future__ import annotations

import numpy as np


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("embedding contains non-finite values")
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("embedding has zero norm")
    return matrix / norms


def top_k(doc_matrix: np.ndarray, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return (row index, cosine score) for the k best rows, best first.

    Scores every row. Ties are broken by corpus order (stable sort), so the
    result is deterministic.
    """
    docs = normalize(doc_matrix)
    query = normalize(query_vector.reshape(1, -1))[0]
    scores = docs @ query
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]
