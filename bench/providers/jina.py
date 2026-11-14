"""Jina AI embeddings API."""

from __future__ import annotations

import httpx
import numpy as np

from bench.providers.base import EmbedResult, Kind, Provider, post_json, require_key

URL = "https://api.jina.ai/v1/embeddings"
TASK = {"document": "retrieval.passage", "query": "retrieval.query"}


class JinaProvider(Provider):
    def __init__(self, run_cfg: dict):
        super().__init__(run_cfg)
        self._key = require_key("JINA_API_KEY")
        self._client = httpx.Client(timeout=180)

    def embed(self, texts: list[str], kind: Kind, model: str) -> EmbedResult:
        data = post_json(
            self._client,
            URL,
            headers={"Authorization": f"Bearer {self._key}"},
            body={"model": model, "task": TASK[kind], "input": texts},
        )
        rows = sorted(data.get("data") or [], key=lambda row: row["index"])
        if len(rows) != len(texts):
            raise RuntimeError(f"Jina returned {len(rows)} embeddings for {len(texts)} inputs")
        usage = data.get("usage") or {}
        if "total_tokens" not in usage:
            raise RuntimeError(f"Jina response has no usage.total_tokens; keys: {sorted(data)}")
        return EmbedResult(
            vectors=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
            tokens=int(usage["total_tokens"]),
            tokens_source="usage.total_tokens",
        )
