"""Harness behaviour that must hold for every provider, tested with a fake one."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import yaml

from bench import data, metrics
from bench.cache import EmbeddingCache
from bench.cost import cost_per_1000_correct, cost_usd
from bench.providers.base import EmbedResult, Provider
from bench.retry import QuotaExhausted, RateLimitError, TransientError, call_with_retry
from bench.run import Embedder, make_batches, rough_tokens
from bench.search import top_k
from bench.throttle import Ledger, RateLimiter


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    monkeypatch.setattr("time.sleep", lambda _: None)


# ---- metrics -------------------------------------------------------------

def test_metrics_hand_computed():
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert metrics.recall_at_k(ranked, {"b", "f", "z"}, 5) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank_at_k(ranked, {"c", "a2"}, 5) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank_at_k(ranked, {"f"}, 5) == 0.0  # rank 6 is outside top 5
    assert metrics.is_correct(ranked, {"e"}, 5) and not metrics.is_correct(ranked, {"f"}, 5)
    with pytest.raises(ValueError):
        metrics.recall_at_k(ranked, set(), 5)


# ---- search --------------------------------------------------------------

def test_search_is_exact_cosine_and_ignores_vector_length():
    docs = np.array([[1, 0], [10, 10], [0, 3], [-1, 0]], dtype=np.float32)
    result = top_k(docs, np.array([2.0, 0.0]), 3)
    assert [i for i, _ in result] == [0, 1, 2]
    assert result[0][1] == pytest.approx(1.0) and result[1][1] == pytest.approx(2**-0.5)


def test_search_breaks_ties_by_corpus_order():
    docs = np.array([[0, 1], [1, 0], [1, 0], [2, 0]], dtype=np.float32)
    assert [i for i, _ in top_k(docs, np.array([1.0, 0.0]), 3)] == [1, 2, 3]


def test_search_rejects_bad_vectors():
    with pytest.raises(ValueError):
        top_k(np.array([[0.0, 0.0]]), np.array([1.0, 0.0]), 1)
    with pytest.raises(ValueError):
        top_k(np.array([[np.nan, 1.0]]), np.array([1.0, 0.0]), 1)


# ---- data integrity ------------------------------------------------------

def test_real_data_matches_manifest():
    ds = data.load_dataset()
    assert len(ds.chunks) == 985 and len(ds.queries) == 30


def test_changed_data_fails_loudly(tmp_path):
    shutil.copytree(data.DATA, tmp_path / "data")
    queries = tmp_path / "data" / "queries.json"
    queries.write_text(queries.read_text(encoding="utf-8").replace("Apple", "Appel", 1), encoding="utf-8")
    with pytest.raises(data.DataMismatchError, match="queries.json"):
        data.load_dataset(tmp_path / "data")


def test_dry_run_subset_is_three_queries_with_their_answers():
    sub = data.dry_run_subset(data.load_dataset())
    ids = {c["chunk_id"] for c in sub.chunks}
    assert len(sub.queries) == 3 and sub.dry_run
    assert all(set(q["relevant_chunk_ids"]) <= ids for q in sub.queries)
    assert len(sub.chunks) < 60


# ---- retry ---------------------------------------------------------------

def test_retry_backs_off_then_succeeds():
    attempts, sleeps = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RateLimitError("429", retry_after=7)
        return "ok"

    result, retries = call_with_retry(flaky, max_attempts=3, sleep=sleeps.append, log=lambda _: None)
    assert (result, retries, len(sleeps)) == ("ok", 2, 2)
    assert all(s >= 7 for s in sleeps)  # honours Retry-After


def test_retry_is_bounded_and_raises_instead_of_dropping():
    calls = []

    def always():
        calls.append(1)
        raise TransientError("503")

    with pytest.raises(RuntimeError, match="gave up after 3"):
        call_with_retry(always, max_attempts=3, sleep=lambda _: None, log=lambda _: None)
    assert len(calls) == 3


@pytest.mark.parametrize("exc", [QuotaExhausted("trial key limit"), KeyError("bad request")])
def test_quota_and_other_errors_are_never_retried(exc):
    calls = []

    def bad():
        calls.append(1)
        raise exc

    with pytest.raises(type(exc)):
        call_with_retry(bad, sleep=lambda _: None)
    assert len(calls) == 1


# ---- rate limits ---------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def limiter(tmp_path=None, **kwargs):
    clock = FakeClock()
    if "ledger" not in kwargs and tmp_path is not None:
        kwargs["ledger"] = Ledger(tmp_path / "ledger.json", month=lambda: "2026-11")
    return clock, RateLimiter(**kwargs, clock=clock, sleep=clock.sleep)


def test_requests_per_minute():
    clock, rl = limiter(rpm=2)
    assert rl.acquire(10) == 0 and rl.acquire(10) == 0
    assert rl.acquire(10) > 59 and clock.now > 59


def test_tokens_per_minute_uses_real_counts_once_known():
    clock, rl = limiter(tpm=1000)
    rl.acquire(100)
    rl.record_tokens(900)  # estimate was too low; the window now holds the real count
    assert rl.acquire(200) > 59
    with pytest.raises(ValueError):
        rl.acquire(5000)


def test_inputs_per_minute():
    _, rl = limiter(ipm=100)
    rl.acquire(1, inputs=96)
    assert rl.acquire(1, inputs=10) > 59


# ---- budgets -------------------------------------------------------------

def test_call_budget_counts_every_attempt_and_persists(tmp_path):
    _, rl = limiter(tmp_path, max_calls_per_month=3)
    for _ in range(3):
        rl.acquire(1)
    _, again = limiter(tmp_path, max_calls_per_month=3)  # a new process
    assert again.ledger.calls_this_month() == 3
    with pytest.raises(QuotaExhausted, match="calls/month"):
        again.acquire(1)


def test_call_budget_resets_with_the_month(tmp_path):
    _, rl = limiter(tmp_path, max_calls_per_month=1)
    rl.acquire(1)
    later = RateLimiter(ledger=Ledger(tmp_path / "ledger.json", month=lambda: "2026-12"), max_calls_per_month=1)
    later.acquire(1)
    assert later.ledger.calls_total == 2


def test_token_budget_counts_real_tokens(tmp_path):
    _, rl = limiter(tmp_path, token_budget=1000)
    rl.acquire(100)
    rl.record_tokens(950)
    with pytest.raises(QuotaExhausted, match="token budget"):
        rl.acquire(100)


# ---- batching ------------------------------------------------------------

def test_batches_respect_count_inputs_and_tokens():
    _, rl = limiter(tpm=1000, rpm=100, ipm=3)
    texts = ["x" * 600] * 7  # 208 rough tokens each; 2 batches fill a minute -> 475 per batch
    batches = make_batches(texts, 10, rl, lambda t: rough_tokens(t[0]))
    assert [len(b) for b in batches] == [2, 2, 2, 1]
    _, free = limiter()
    assert [len(b) for b in make_batches(["a"] * 7, 3, free, lambda t: 1)] == [3, 3, 1]


# ---- cache + embedder ----------------------------------------------------

class FakeProvider(Provider):
    """Deterministic vectors; counts calls and can fail on demand."""

    def __init__(self, fail_first: int = 0, error=RateLimitError):
        super().__init__({})
        self.calls: list[tuple[str, str, int]] = []
        self.embedded: list[str] = []
        self.fail_first, self.error = fail_first, error

    def embed(self, texts, kind, model):
        if self.fail_first:
            self.fail_first -= 1
            raise self.error("429")
        self.calls.append((kind, model, len(texts)))
        self.embedded += texts
        vectors = np.array([[len(t) + 1.0, sum(map(ord, t)) % 97 + 1, 1.0] for t in texts], dtype=np.float32)
        return EmbedResult(vectors=vectors, tokens=sum(len(t.split()) for t in texts), tokens_source="fake")


def embedder(tmp_path, provider, *, max_attempts=3, **limits):
    _, rl = limiter(tmp_path, **limits)
    return Embedder(provider, EmbeddingCache(tmp_path / "c.sqlite", "fake:1"), rl, max_attempts=max_attempts)


def test_rerun_uses_cache_and_keeps_measured_numbers(tmp_path):
    texts = ["alpha beta", "gamma", "alpha beta", "delta epsilon zeta"]

    first = FakeProvider()
    e1 = embedder(tmp_path, first)
    docs1 = e1.embed_documents("m", texts, batch_texts=2)
    q1 = e1.embed_queries("m", ["what is alpha"])
    # 3 unique texts in batches of 2 -> 2 calls; then warm-up + 1 query.
    assert first.calls == [("document", "m", 2), ("document", "m", 1), ("query", "m", 1), ("query", "m", 1)]
    assert q1[0][2] is False

    second = FakeProvider()
    e2 = embedder(tmp_path, second)
    docs2 = e2.embed_documents("m", texts, batch_texts=2)
    q2 = e2.embed_queries("m", ["what is alpha"])
    assert second.calls == []  # nothing re-embedded, nothing re-billed
    assert e2.limiter.ledger.calls_total == 4  # only the first run's calls
    assert q2[0][2] is True
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(docs1, docs2))
    assert e2.cache.call(q2[0][1])["tokens"] == 3
    assert np.array_equal(docs2[0][0], docs2[2][0])  # duplicate text, same vector


def test_plan_counts_only_uncached_work(tmp_path):
    e = embedder(tmp_path, FakeProvider())
    cfg = {"index_model": "m", "query_model": "m", "batch_texts": 2}
    assert e.plan(cfg, ["a", "b", "c"], ["q1", "q2"])["calls"] == 2 + 2 + 1
    e.embed_documents("m", ["a", "b", "c"], batch_texts=2)
    e.embed_queries("m", ["q1", "q2"])
    assert e.plan(cfg, ["a", "b", "c"], ["q1", "q2"]) == {"chunks": 0, "queries": 0, "calls": 0, "tokens": 0}


def test_cache_separates_models_and_kinds(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite", "fake:1")
    cache.put_call("pro", "document", ["x"], np.ones((1, 2)), tokens=1, tokens_source="t",
                   elapsed_s=0.1, wait_s=0, retries=0)
    assert cache.get("pro", "document", "x") is not None
    assert cache.get("fast", "document", "x") is None
    assert cache.get("pro", "query", "x") is None


def test_rate_limited_call_is_retried_and_each_attempt_counted(tmp_path):
    e = embedder(tmp_path, FakeProvider(fail_first=2), max_attempts=3)
    out = e.embed_queries("m", ["q one", "q two"])
    assert len(out) == 2
    assert e.limiter.ledger.calls_total == 2 + 3  # 2 failed attempts + warm-up + 2 queries
    assert e.cache.call(out[0][1])["retries"] == 0


def test_bounded_retries_stop_the_run(tmp_path):
    e = embedder(tmp_path, FakeProvider(fail_first=5), max_attempts=2)
    with pytest.raises(RuntimeError, match="gave up after 2"):
        e.embed_documents("m", ["a"], batch_texts=1)
    assert e.limiter.ledger.calls_total == 2


def test_budget_stop_keeps_progress_and_resumes(tmp_path):
    texts = [f"chunk {i}" for i in range(5)]
    provider = FakeProvider()
    with pytest.raises(QuotaExhausted):
        embedder(tmp_path, provider, max_calls_per_month=3).embed_documents("m", texts, batch_texts=1)
    assert provider.embedded == texts[:3]
    # Rerun with room in the budget: only the two missing chunks are embedded.
    out = embedder(tmp_path, provider, max_calls_per_month=10).embed_documents("m", texts, batch_texts=1)
    assert provider.embedded == texts and all(v is not None for v in out)


# ---- cost + config -------------------------------------------------------

def test_cost_arithmetic():
    assert cost_usd(2_000_000, 0.12) == pytest.approx(0.24)
    assert cost_per_1000_correct(0.0001, 20) == pytest.approx(0.005)
    assert cost_per_1000_correct(0.0001, 0) is None


def test_config_has_required_fields():
    cfg = yaml.safe_load((data.ROOT / "config" / "providers.yaml").read_text(encoding="utf-8"))
    assert cfg["checked"] and cfg["top_k"] == 5
    assert set(cfg["runs"]) == {"local-bge-m3", "local-minilm", "cohere-pro", "cohere-pro-fast",
                                "voyage-4-large", "jina-v5-small"}
    for name, run in cfg["runs"].items():
        for field in ("adapter", "label", "index_model", "query_model", "index_price_per_1m",
                      "query_price_per_1m", "price_source", "batch_texts", "limits", "sources"):
            assert field in run, f"{name} missing {field}"
        assert set(run["limits"]) == {"rpm", "tpm", "ipm"}


def test_new_process_inherits_last_minute_of_calls(tmp_path):
    first = RateLimiter(tpm=1000, ledger=Ledger(tmp_path / "l.json"))
    first.acquire(100)
    first.record_tokens(900)
    clock = FakeClock()
    second = RateLimiter(tpm=1000, ledger=Ledger(tmp_path / "l.json"), clock=clock, sleep=clock.sleep)
    assert second.acquire(200) > 50  # waits out the earlier process's tokens
