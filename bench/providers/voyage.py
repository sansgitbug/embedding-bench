"""Voyage AI embeddings API."""

from __future__ import annotations

import httpx
import numpy as np

from bench.providers.base import EmbedResult, Kind, Provider, post_json, require_key

URL = "https://api.voyageai.com/v1/embeddings"


class VoyageProvider(Provider):
    def __init__(self, run_cfg: dict):
        super().__init__(run_cfg)
        self._key = require_key("VOYAGE_API_KEY")
        self._client = httpx.Client(timeout=120)

    def embed(self, texts: list[str], kind: Kind, model: str) -> EmbedResult:
        data = post_json(
            self._client,
            URL,
            headers={"Authorization": f"Bearer {self._key}"},
            body={
                "model": model,
                "input": texts,
                "input_type": kind,  # "document" or "query"
                "truncation": False,  # error rather than silently cut a chunk
            },
        )
        rows = sorted(data["data"], key=lambda row: row["index"])
        return EmbedResult(
            vectors=np.asarray([row["embedding"] for row in rows], dtype=np.float32),
            tokens=int(data["usage"]["total_tokens"]),
            tokens_source="usage.total_tokens",
        )
