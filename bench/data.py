"""Load the benchmark data and refuse to run if it differs from the manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DRY_RUN_QUERIES = 3
DRY_RUN_EXTRA_CHUNKS = 40


class DataMismatchError(RuntimeError):
    """The data on disk is not the data the benchmark was defined on."""


@dataclass(frozen=True)
class Dataset:
    chunks: list[dict]
    queries: list[dict]
    near_duplicates: dict[str, list[dict]]
    corpus_sha256: str
    queries_sha256: str
    dry_run: bool = False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(data_dir: Path = DATA) -> Dataset:
    manifest = json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        actual = _sha256(data_dir / name)
        if actual != expected:
            raise DataMismatchError(
                f"{name} has sha256 {actual} but MANIFEST.json expects {expected}. "
                "Every provider must see identical chunks and queries; restore the file."
            )

    with (data_dir / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    queries = json.loads((data_dir / "queries.json").read_text(encoding="utf-8"))
    near = json.loads((data_dir / "near_duplicates.json").read_text(encoding="utf-8"))

    if len(chunks) != manifest["chunk_count"] or len(queries) != manifest["query_count"]:
        raise DataMismatchError("chunk or query count differs from MANIFEST.json")
    ids = {c["chunk_id"] for c in chunks}
    if len(ids) != len(chunks):
        raise DataMismatchError("duplicate chunk IDs")
    for q in queries:
        if not q["relevant_chunk_ids"] or not set(q["relevant_chunk_ids"]) <= ids:
            raise DataMismatchError(f"{q['query_id']} has missing or empty relevant chunks")

    return Dataset(
        chunks=chunks,
        queries=queries,
        near_duplicates=near["by_query"],
        corpus_sha256=manifest["sha256"]["chunks.jsonl"],
        queries_sha256=manifest["sha256"]["queries.json"],
    )


def dry_run_subset(ds: Dataset) -> Dataset:
    """First 3 queries over their relevant chunks plus the first 40 chunks.

    Deterministic, so every provider's dry run sees the same small problem.
    """
    queries = ds.queries[:DRY_RUN_QUERIES]
    wanted = {cid for q in queries for cid in q["relevant_chunk_ids"]}
    wanted |= {c["chunk_id"] for c in ds.chunks[:DRY_RUN_EXTRA_CHUNKS]}
    chunks = [c for c in ds.chunks if c["chunk_id"] in wanted]
    return Dataset(
        chunks=chunks,
        queries=queries,
        near_duplicates=ds.near_duplicates,
        corpus_sha256=ds.corpus_sha256,
        queries_sha256=ds.queries_sha256,
        dry_run=True,
    )
