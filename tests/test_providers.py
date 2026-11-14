"""API adapters with the HTTP layer mocked: request shape, parsing, and failure mapping."""

from __future__ import annotations

import json

import httpx
import pytest

from bench.providers.cohere import CohereProvider
from bench.providers.jina import JinaProvider
from bench.providers.voyage import VoyageProvider
from bench.retry import QuotaExhausted, RateLimitError, TransientError


def mocked(monkeypatch, cls, env, handler):
    monkeypatch.setenv(env, "test-key")
    sent = []

    def record(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append((str(request.url), request.headers["authorization"], body))
        return handler(body)

    provider = cls({})
    provider._client = httpx.Client(transport=httpx.MockTransport(record))
    return provider, sent


def openai_style(body, n_key="input"):
    n = len(body[n_key])
    # Deliberately out of order: adapters must sort by index.
    rows = [{"index": i, "embedding": [float(i + 1), 1.0]} for i in reversed(range(n))]
    return httpx.Response(200, json={"data": rows, "usage": {"total_tokens": 7 * n}})


def test_jina_request_and_response(monkeypatch):
    provider, sent = mocked(monkeypatch, JinaProvider, "JINA_API_KEY", openai_style)
    result = provider.embed(["a", "b"], "document", "jina-embeddings-v5-text-small")
    url, auth, body = sent[0]
    assert url == "https://api.jina.ai/v1/embeddings" and auth == "Bearer test-key"
    assert body == {"model": "jina-embeddings-v5-text-small", "task": "retrieval.passage", "input": ["a", "b"]}
    assert result.vectors[:, 0].tolist() == [1.0, 2.0] and result.tokens == 14
    provider.embed(["q"], "query", "jina-embeddings-v5-text-small")
    assert sent[1][2]["task"] == "retrieval.query"


def test_voyage_request_and_response(monkeypatch):
    provider, sent = mocked(monkeypatch, VoyageProvider, "VOYAGE_API_KEY", openai_style)
    result = provider.embed(["a", "b", "c"], "query", "voyage-4-large")
    assert sent[0][2] == {"model": "voyage-4-large", "input": ["a", "b", "c"], "input_type": "query",
                          "truncation": False}
    assert result.vectors[:, 0].tolist() == [1.0, 2.0, 3.0] and result.tokens == 21


def test_cohere_request_and_response(monkeypatch):
    provider, sent = mocked(monkeypatch, CohereProvider, "COHERE_API_KEY", lambda body: httpx.Response(200, json={
        "embeddings": {"float": [[1.0, 0.0]] * len(body["texts"])},
        "meta": {"billed_units": {"input_tokens": 9}},
    }))
    result = provider.embed(["a", "b"], "document", "embed-v5.0-pro")
    assert sent[0][2]["input_type"] == "search_document" and sent[0][2]["truncate"] == "NONE"
    assert result.vectors.shape == (2, 2) and result.tokens == 9


def test_wrong_vector_count_fails(monkeypatch):
    provider, _ = mocked(monkeypatch, JinaProvider, "JINA_API_KEY", lambda body: httpx.Response(
        200, json={"data": [{"index": 0, "embedding": [1.0]}], "usage": {"total_tokens": 1}}))
    with pytest.raises(RuntimeError, match="1 embeddings for 2 inputs"):
        provider.embed(["a", "b"], "document", "m")


def test_missing_usage_fails(monkeypatch):
    provider, _ = mocked(monkeypatch, JinaProvider, "JINA_API_KEY", lambda body: httpx.Response(
        200, json={"data": [{"index": 0, "embedding": [1.0]}]}))
    with pytest.raises(RuntimeError, match="usage.total_tokens"):
        provider.embed(["a"], "document", "m")


@pytest.mark.parametrize("status, text, expected", [
    (429, "Too many requests, slow down", RateLimitError),
    (429, "You are using a Trial key, which is limited to 1000 API calls / month.", QuotaExhausted),
    (402, "Insufficient balance", QuotaExhausted),
    (503, "unavailable", TransientError),
    (400, "bad input", RuntimeError),
])
def test_http_failures_map_to_retry_or_stop(monkeypatch, status, text, expected):
    provider, _ = mocked(monkeypatch, JinaProvider, "JINA_API_KEY", lambda body: httpx.Response(status, text=text))
    with pytest.raises(expected):
        provider.embed(["a"], "document", "m")


def test_per_minute_429_waits_a_full_minute(monkeypatch):
    provider, _ = mocked(monkeypatch, CohereProvider, "COHERE_API_KEY", lambda body: httpx.Response(
        429, text='{"message":"trial token rate limit exceeded, limit is 100000 tokens per minute"}'))
    with pytest.raises(RateLimitError) as exc:
        provider.embed(["a"], "document", "embed-v5.0-pro")
    assert exc.value.retry_after >= 60
