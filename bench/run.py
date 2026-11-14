"""Run the benchmark for one or more provider setups.

    python -m bench.run --estimate                 # tokens, calls and list-price cost still to spend
    python -m bench.run --provider jina-v5-small --dry-run
    python -m bench.run --provider jina-v5-small   # asks before a full API run
    python -m bench.run --all --yes
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from bench import cost
from bench.cache import EmbeddingCache
from bench.data import ROOT, Dataset, dry_run_subset, load_dataset
from bench.providers.base import Provider
from bench.retry import QuotaExhausted, call_with_retry
from bench.search import top_k
from bench.throttle import Ledger, RateLimiter

CONFIG = ROOT / "config" / "providers.yaml"
CACHE_DIR = ROOT / "cache"
ADAPTERS = {
    "local": ("bench.providers.local", "LocalProvider"),
    "cohere": ("bench.providers.cohere", "CohereProvider"),
    "voyage": ("bench.providers.voyage", "VoyageProvider"),
    "jina": ("bench.providers.jina", "JinaProvider"),
}
CACHE_VERSION = "1"


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def make_provider(run_cfg: dict) -> Provider:
    module, cls = ADAPTERS[run_cfg["adapter"]]
    return getattr(importlib.import_module(module), cls)(run_cfg)


def open_cache(run_cfg: dict) -> EmbeddingCache:
    adapter = run_cfg["adapter"]
    return EmbeddingCache(CACHE_DIR / f"{adapter}.sqlite", namespace=f"{adapter}:{CACHE_VERSION}")


def make_limiter(run_cfg: dict) -> RateLimiter:
    api = run_cfg["adapter"] != "local"
    return RateLimiter(
        **run_cfg["limits"],
        ledger=Ledger(CACHE_DIR / f"{run_cfg['adapter']}.ledger.json") if api else None,
        max_calls_per_month=run_cfg.get("max_calls_per_month"),
        token_budget=run_cfg.get("token_budget"),
    )


def rough_tokens(text: str) -> int:
    """Deliberately high guess (about 1.6x real counts on this corpus)."""
    return len(text) // 3 + 8


def batch_token_budget(limiter: RateLimiter) -> int | None:
    """Tokens per batch so that two batches fill one minute's token allowance."""
    if limiter.tpm is None:
        return None
    return int(0.95 * limiter.tpm / min(limiter.rpm or 2, 2))


def make_batches(texts: list[str], batch_texts: int, limiter: RateLimiter, estimate) -> list[list[str]]:
    cap = min(batch_texts, limiter.ipm or batch_texts)
    budget = batch_token_budget(limiter)
    batches: list[list[str]] = []
    batch: list[str] = []
    used = 0
    for text in texts:
        need = estimate([text])
        if batch and (len(batch) >= cap or (budget is not None and used + need > budget)):
            batches.append(batch)
            batch, used = [], 0
        batch.append(text)
        used += need
    if batch:
        batches.append(batch)
    return batches


class Embedder:
    """Cache -> budget -> throttle -> bounded retry -> provider. The same path for every provider."""

    def __init__(self, provider: Provider, cache: EmbeddingCache, limiter: RateLimiter, *, max_attempts: int = 3):
        self.provider, self.cache, self.limiter = provider, cache, limiter
        self.max_attempts = max_attempts
        self._rough_seen = 0
        self._actual_seen = 0

    def estimate(self, texts: list[str]) -> int:
        """Token estimate for throttling, calibrated on this provider's real counts once available."""
        rough = sum(rough_tokens(t) for t in texts)
        if self._rough_seen < 5000:
            return rough
        ratio = min(1.0, self._actual_seen / self._rough_seen * 1.15)
        return int(rough * ratio) + 1

    def _call(self, model: str, kind: str, texts: list[str], *, store: bool = True) -> None:
        est = self.estimate(texts)
        waits: list[float] = []
        timings: list[float] = []

        def attempt():
            waits.append(self.limiter.acquire(est, len(texts)))
            started = time.perf_counter()
            try:
                return self.provider.embed(texts, kind, model)
            finally:
                timings.append(time.perf_counter() - started)

        result, retries = call_with_retry(attempt, max_attempts=self.max_attempts)
        self.limiter.record_tokens(result.tokens)
        self._rough_seen += sum(rough_tokens(t) for t in texts)
        self._actual_seen += result.tokens
        if result.vectors.shape[0] != len(texts):
            raise RuntimeError(f"asked for {len(texts)} embeddings, got {result.vectors.shape[0]}")
        if store:
            self.cache.put_call(
                model, kind, texts, result.vectors, tokens=result.tokens,
                tokens_source=result.tokens_source, elapsed_s=timings[-1],
                wait_s=sum(waits), retries=retries,
            )

    def todo(self, index_model: str, chunk_texts: list[str], query_model: str, query_texts: list[str]):
        docs = list(dict.fromkeys(t for t in chunk_texts if self.cache.get(index_model, "document", t) is None))
        queries = [t for t in query_texts if self.cache.get(query_model, "query", t) is None]
        return docs, queries

    def plan(self, run_cfg: dict, chunk_texts: list[str], query_texts: list[str]) -> dict:
        """Calls and (pessimistic) tokens still needed; cached texts cost nothing."""
        docs, queries = self.todo(run_cfg["index_model"], chunk_texts, run_cfg["query_model"], query_texts)
        batches = make_batches(docs, run_cfg["batch_texts"], self.limiter, lambda t: rough_tokens(t[0]))
        warmup = 1 if queries else 0
        return {
            "chunks": len(docs),
            "queries": len(queries),
            "calls": len(batches) + len(queries) + warmup,
            "tokens": sum(map(rough_tokens, docs)) + sum(map(rough_tokens, queries)) + warmup * 10,
        }

    def embed_documents(self, model: str, texts: list[str], batch_texts: int) -> list[tuple]:
        missing = list(dict.fromkeys(t for t in texts if self.cache.get(model, "document", t) is None))
        if missing:
            print(f"  embedding {len(missing)} chunks ({len(texts) - len(missing)} already cached)", flush=True)
        done = 0
        started = time.monotonic()
        # Batches are re-planned as the token estimate calibrates.
        while missing:
            batch = make_batches(missing, batch_texts, self.limiter, self.estimate)[0]
            self._call(model, "document", batch)
            missing = missing[len(batch):]
            done += len(batch)
            rate = done / max(time.monotonic() - started, 1e-9)
            eta = len(missing) / rate / 60 if rate else math.inf
            print(f"    {done} done, {len(missing)} left, ~{eta:.0f} min to go", flush=True)
        return [self.cache.get(model, "document", t) for t in texts]

    def embed_queries(self, model: str, texts: list[str]) -> list[tuple]:
        """One query per call, sequentially, so each latency is one real request."""
        fresh = {t for t in texts if self.cache.get(model, "query", t) is None}
        if fresh:
            # Untimed warm-up so connection setup or model load is not billed to query 1.
            self._call(model, "query", ["warm-up"], store=False)
        for text in texts:
            if text in fresh:
                self._call(model, "query", [text])
        return [(*self.cache.get(model, "query", t), t not in fresh) for t in texts]


def ledger_line(limiter: RateLimiter) -> str:
    led = limiter.ledger
    if led is None:
        return ""
    parts = [f"{led.calls_total} API calls ({led.calls_this_month()} this month)", f"{led.tokens_total:,} tokens"]
    if limiter.max_calls_per_month:
        parts[0] += f" of {limiter.max_calls_per_month}/month budget"
    if limiter.token_budget:
        parts[1] += f" of {limiter.token_budget:,} budget"
    return "  used so far by this harness: " + ", ".join(parts)


def run_one(name: str, run_cfg: dict, ds: Dataset, top: int, checked: str) -> None:
    print(f"\n== {name}: {run_cfg['label']}" + (" [DRY RUN]" if ds.dry_run else ""), flush=True)
    limiter = make_limiter(run_cfg)
    cache = open_cache(run_cfg)
    chunk_texts = [c["text"] for c in ds.chunks]
    query_texts = [q["query"] for q in ds.queries]

    # Check budgets before making a single call, with a pessimistic token guess.
    plan = Embedder(None, cache, limiter).plan(run_cfg, chunk_texts, query_texts)  # type: ignore[arg-type]
    print(f"  to embed: {plan['chunks']} chunks and {plan['queries']} queries, "
          f"~{plan['calls']} calls, at most ~{plan['tokens']:,} tokens", flush=True)
    try:
        limiter.check_budget(plan["calls"], plan["tokens"])
    except QuotaExhausted as exc:
        raise SystemExit(f"{name}: not started. {exc}") from exc

    provider = make_provider(run_cfg)
    embedder = Embedder(provider, cache, limiter, max_attempts=run_cfg.get("max_attempts", 3))
    marker = CACHE_DIR / f"{name}.running"
    if not ds.dry_run:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    try:
        docs = embedder.embed_documents(run_cfg["index_model"], chunk_texts, run_cfg["batch_texts"])
        queries = embedder.embed_queries(run_cfg["query_model"], query_texts)
    except (QuotaExhausted, RuntimeError) as exc:
        print(ledger_line(limiter))
        raise SystemExit(f"{name}: stopped, no results written; finished work is cached. {exc}") from exc
    finally:
        marker.unlink(missing_ok=True)

    doc_matrix = np.vstack([vec for vec, _ in docs])
    index_calls = [cache.call(cid) for cid in sorted({cid for _, cid in docs})]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = []
    for q, (vec, call_id, was_cached) in zip(ds.queries, queries, strict=True):
        call = cache.call(call_id)
        lines.append({
            "run": name,
            "query_id": q["query_id"],
            "query": q["query"],
            "relevant_chunk_ids": q["relevant_chunk_ids"],
            "top_k": [
                {"chunk_id": ds.chunks[i]["chunk_id"], "score": round(score, 6)}
                for i, score in top_k(doc_matrix, vec, top)
            ],
            "latency_ms": round(call["elapsed_s"] * 1000, 3),
            "tokens": call["tokens"],
            "tokens_source": call["tokens_source"],
            "retries": call["retries"],
            "query_model": run_cfg["query_model"],
            "embedding_from_cache": was_cached,
            "measured_at": datetime.fromtimestamp(call["created_at"], timezone.utc).isoformat(timespec="seconds"),
            "corpus_sha256": ds.corpus_sha256,
            "queries_sha256": ds.queries_sha256,
        })
    if len(lines) != len(ds.queries):
        raise RuntimeError("a query was dropped")  # cannot happen; guards the invariant

    meta = {
        "run": name,
        "label": run_cfg["label"],
        "dry_run": ds.dry_run,
        "index_model": run_cfg["index_model"],
        "query_model": run_cfg["query_model"],
        "dimensions": int(doc_matrix.shape[1]),
        "chunks": len(ds.chunks),
        "queries": len(ds.queries),
        "top_k": top,
        "index_tokens": sum(c["tokens"] for c in index_calls),
        "index_seconds_in_embedding_calls": round(sum(c["elapsed_s"] for c in index_calls), 3),
        "index_seconds_waiting_on_rate_limits": round(sum(c["wait_s"] for c in index_calls), 3),
        "index_calls": len(index_calls),
        "index_retries": sum(c["retries"] for c in index_calls),
        "index_price_per_1m": run_cfg["index_price_per_1m"],
        "query_price_per_1m": run_cfg["query_price_per_1m"],
        "price_source": run_cfg["price_source"],
        "prices_checked": checked,
        "limits": run_cfg["limits"],
        "corpus_sha256": ds.corpus_sha256,
        "queries_sha256": ds.queries_sha256,
        "written_at": now,
        "platform": platform.platform(),
        "python": platform.python_version(),
        **provider.describe(),
    }
    cache.close()

    out_dir = ROOT / "results" / ("dry-run" if ds.dry_run else "raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{name}.jsonl.tmp"
    tmp.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    tmp.replace(out_dir / f"{name}.jsonl")
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    hits = sum(
        bool({t["chunk_id"] for t in line["top_k"]} & set(line["relevant_chunk_ids"])) for line in lines
    )
    print(f"  {hits}/{len(lines)} queries with a relevant chunk in the top {top}; "
          f"wrote {(out_dir / (name + '.jsonl')).relative_to(ROOT)}")
    if limiter.ledger is not None:
        print(ledger_line(limiter))


def print_estimate(names: list[str], cfg: dict, ds: Dataset) -> None:
    print(f"\nStill to spend for {len(ds.chunks)} chunks and {len(ds.queries)} queries "
          f"(cached texts are free; tokens are a deliberately high guess; list prices checked {cfg['checked']})")
    print(f"{'run':<18}{'chunks':>8}{'queries':>9}{'calls':>7}{'~tokens (max)':>15}{'list cost':>11}")
    for name in names:
        run_cfg = cfg["runs"][name]
        limiter = make_limiter(run_cfg)
        cache = open_cache(run_cfg)
        p = Embedder(None, cache, limiter).plan(  # type: ignore[arg-type]
            run_cfg, [c["text"] for c in ds.chunks], [q["query"] for q in ds.queries])
        cache.close()
        price = run_cfg["index_price_per_1m"]
        print(f"{name:<18}{p['chunks']:>8}{p['queries']:>9}{p['calls']:>7}{p['tokens']:>15,}"
              f"{'$' + format(cost.cost_usd(p['tokens'], price), '.4f'):>11}")
        line = ledger_line(limiter)
        if line:
            print(line)
    print("cohere-pro-fast reuses cohere-pro's chunk embeddings, so once cohere-pro has run "
          "it needs only its query calls.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", action="append", help="run name from config/providers.yaml (repeatable)")
    parser.add_argument("--all", action="store_true", help="every run in the config")
    parser.add_argument("--dry-run", action="store_true", help="3 queries over a 40-odd chunk subset")
    parser.add_argument("--estimate", action="store_true", help="print what is still to spend and exit")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation before a full API run")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = load_config()
    names = list(cfg["runs"]) if args.all or (args.estimate and not args.provider) else args.provider or []
    unknown = [n for n in names if n not in cfg["runs"]]
    if unknown or not names:
        parser.error(f"choose --all or --provider from: {', '.join(cfg['runs'])}")

    ds = load_dataset()  # raises if the data differs from the manifest
    if args.dry_run:
        ds = dry_run_subset(ds)

    uses_api = any(cfg["runs"][n]["adapter"] != "local" for n in names)
    if args.estimate or (uses_api and not args.dry_run):
        print_estimate(names, cfg, ds)
        if args.estimate:
            return
        if not args.yes:
            if not sys.stdin.isatty():
                raise SystemExit("Full API run needs confirmation: rerun with --yes.")
            if input("Proceed with the full run? [y/N] ").strip().lower() != "y":
                raise SystemExit("Cancelled.")

    for name in names:
        run_one(name, cfg["runs"][name], ds, cfg["top_k"], cfg["checked"])


if __name__ == "__main__":
    main()
