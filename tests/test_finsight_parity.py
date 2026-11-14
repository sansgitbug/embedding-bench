"""Check the ported metrics agree with FinSight's original implementation.

Skipped unless FINSIGHT_PATH points at a FinSight checkout. FinSight's module
imports its vector store (FAISS, Chroma), so a stub stands in for that import;
only FinSight's own metric functions are exercised.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from bench import metrics

FINSIGHT = os.environ.get("FINSIGHT_PATH")


@dataclass
class _Chunk:
    chunk_id: str


@dataclass
class _SearchResult:
    chunk: _Chunk


@pytest.mark.skipif(not FINSIGHT, reason="set FINSIGHT_PATH to run")
def test_metrics_match_finsight(monkeypatch):
    stub = types.ModuleType("src.retrieval.vectorstore")
    stub.SearchResult = _SearchResult
    for name in ("src", "src.retrieval"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "src.retrieval.vectorstore", stub)

    path = Path(FINSIGHT) / "src" / "evaluation" / "retrieval_metrics.py"
    spec = importlib.util.spec_from_file_location("finsight_metrics", path)
    original = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "finsight_metrics", original)
    spec.loader.exec_module(original)

    rng = random.Random(0)
    pool = [f"c{i}" for i in range(12)]
    for _ in range(500):
        ranked = rng.sample(pool, rng.randint(0, 8))
        relevant = set(rng.sample(pool, rng.randint(1, 4)))
        results = [_SearchResult(_Chunk(cid)) for cid in ranked]
        assert metrics.recall_at_k(ranked, relevant, 5) == original.recall_at_k(results, relevant, 5)
        assert metrics.reciprocal_rank_at_k(ranked, relevant, 5) == original.reciprocal_rank(results, relevant, 5)
