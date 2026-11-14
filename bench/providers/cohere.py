"""Cohere Embed v2 API."""

from __future__ import annotations

import httpx
import numpy as np

from bench.providers.base import EmbedResult, Kind, Provider, post_json, require_key

URL = "https://api.cohere.com/v2/embed"
INPUT_TYPE = {"document": "search_document", "query": "search_query"}


class CohereProvider(Provider):
    def __init__(self, run_cfg: dict):
        super().__init__(run_cfg)
        self._key = require_key("COHERE_API_KEY")
        self._client = httpx.Client(timeout=120)

    def embed(self, texts: list[str], kind: Kind, model: str) -> EmbedResult:
        data = post_json(
            self._client,
            URL,
            headers={"Authorization": f"Bearer {self._key}"},
            body={
                "model": model,
                "texts": texts,
                "input_type": INPUT_TYPE[kind],
                "embedding_types": ["float"],
                "truncate": "NONE",  # error rather than silently cut a chunk
            },
        )
        return EmbedResult(
            vectors=np.asarray(data["embeddings"]["float"], dtype=np.float32),
            tokens=int(data["meta"]["billed_units"]["input_tokens"]),
            tokens_source="meta.billed_units.input_tokens",
        )
