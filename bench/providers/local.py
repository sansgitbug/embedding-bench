"""Local open-source models via sentence-transformers. Free baseline."""

from __future__ import annotations

from bench.providers.base import EmbedResult, Kind, Provider


class LocalProvider(Provider):
    def __init__(self, run_cfg: dict):
        super().__init__(run_cfg)
        self._models: dict = {}

    def _model(self, name: str):
        if name not in self._models:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(name)
            # Never read more than the configured window. The model's own limit
            # may be lower still (all-MiniLM-L6-v2 stops at 256 tokens).
            model.max_seq_length = min(model.max_seq_length, self.cfg["max_input_tokens"])
            self._models[name] = model
        return self._models[name]

    def embed(self, texts: list[str], kind: Kind, model: str) -> EmbedResult:
        st = self._model(model)
        vectors = st.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Tokens the model actually read, i.e. after truncation.
        encoded = st.tokenizer(texts, truncation=True, max_length=st.max_seq_length)
        tokens = sum(len(ids) for ids in encoded["input_ids"])
        return EmbedResult(
            vectors=vectors, tokens=tokens, tokens_source="local tokenizer, after truncation"
        )

    def describe(self) -> dict:
        import sentence_transformers
        import torch

        return {
            "device": str(next(iter(self._models.values())).device) if self._models else None,
            "max_seq_length": {name: m.max_seq_length for name, m in self._models.items()},
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        }
