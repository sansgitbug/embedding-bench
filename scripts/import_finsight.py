"""One-off import of chunks and queries from a local FinSight checkout.

Usage: python scripts/import_finsight.py <path-to-FinSight>

Writes data/chunks.jsonl, data/queries.json, data/near_duplicates.json and
data/MANIFEST.json. The outputs are committed, so nobody reproducing the
benchmark needs FinSight or needs to run this script.
"""

from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# A chunk counts as a near-duplicate of a labelled chunk when the Jaccard
# overlap of their 8-word shingles exceeds this. Fixed before any provider ran.
NEAR_DUP_THRESHOLD = 0.8
SHINGLE_WORDS = 8


def shingles(text: str) -> set[str]:
    words = text.split()
    last = max(1, len(words) - SHINGLE_WORDS + 1)
    return {" ".join(words[i : i + SHINGLE_WORDS]) for i in range(last)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(finsight: Path) -> None:
    chunks = []
    for file in sorted(glob.glob(str(finsight / "chunks" / "*" / "*" / "chunks.json"))):
        for c in json.loads(Path(file).read_text(encoding="utf-8")):
            meta = c.get("metadata", {})
            chunks.append(
                {
                    "chunk_id": c["chunk_id"],
                    "ticker": c["ticker"],
                    "company_name": meta.get("company_name"),
                    "filing_type": c["filing_type"],
                    "filing_date": c["filing_date"],
                    "section_name": c["section_name"],
                    "accession_number": meta.get("accession_number"),
                    "source_url": meta.get("source_url"),
                    # Token count from FinSight's chunker (bert-base-uncased).
                    "token_count": c["token_count"],
                    "text": c["text"],
                }
            )
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate chunk IDs in FinSight chunks")

    raw_queries = json.loads(
        (finsight / "data" / "evaluation_queries_v2.json").read_text(encoding="utf-8")
    )
    queries = [
        {
            "query_id": f"q{i:02d}",
            "query": q["query"],
            "ticker": q["ticker"],
            "relevant_chunk_ids": q["relevant_chunk_ids"],
        }
        for i, q in enumerate(raw_queries, start=1)
    ]
    known = set(ids)
    for q in queries:
        missing = [r for r in q["relevant_chunk_ids"] if r not in known]
        if missing:
            raise SystemExit(f"{q['query_id']}: relevant chunks not in corpus: {missing}")

    by_id = {c["chunk_id"]: shingles(c["text"]) for c in chunks}
    near: dict[str, list[dict]] = {}
    for q in queries:
        labelled = set(q["relevant_chunk_ids"])
        found = {}
        for rel in sorted(labelled):
            for cid, sh in by_id.items():
                if cid in labelled:
                    continue
                union = len(by_id[rel] | sh)
                jaccard = len(by_id[rel] & sh) / union if union else 0.0
                if jaccard > NEAR_DUP_THRESHOLD and jaccard > found.get(cid, (0, ""))[0]:
                    found[cid] = (jaccard, rel)
        near[q["query_id"]] = [
            {"chunk_id": cid, "near_duplicate_of": rel, "jaccard": round(j, 3)}
            for cid, (j, rel) in sorted(found.items())
        ]

    DATA.mkdir(exist_ok=True)
    with (DATA / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (DATA / "queries.json").write_text(
        json.dumps(queries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    (DATA / "near_duplicates.json").write_text(
        json.dumps(
            {
                "method": f"Jaccard overlap of {SHINGLE_WORDS}-word shingles > {NEAR_DUP_THRESHOLD} "
                "with a labelled relevant chunk, excluding the labelled chunks themselves",
                "by_query": near,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "source": "FinSight (https://github.com/sansgitbug/Finsight), SEC EDGAR 10-K and 10-Q filings",
        "chunk_count": len(chunks),
        "query_count": len(queries),
        "sha256": {
            name: sha256(DATA / name)
            for name in ("chunks.jsonl", "queries.json", "near_duplicates.json")
        },
    }
    (DATA / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    with_near = sum(1 for v in near.values() if v)
    print(f"{len(chunks)} chunks, {len(queries)} queries, {with_near} queries with near-duplicates")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
