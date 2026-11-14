"""The report must refuse results that were not produced from the same data."""

from __future__ import annotations

import json

import pytest

from bench import report
from bench.data import load_dataset


def fake_run(ds, tmp_path, name="fake", *, hit_ids=("q01",), mutate=None, top=None):
    lines = []
    for q in ds.queries:
        first = q["relevant_chunk_ids"][0] if q["query_id"] in hit_ids else "MSFT-2025-04-30-0001"
        ids = top.get(q["query_id"], [first]) if top else [first]
        ids = (ids + ["NVDA-2025-02-26-0001", "NVDA-2025-02-26-0002", "NVDA-2025-02-26-0003",
                      "NVDA-2025-02-26-0004"])[:5]
        lines.append({
            "query_id": q["query_id"], "query": q["query"],
            "relevant_chunk_ids": q["relevant_chunk_ids"],
            "top_k": [{"chunk_id": c, "score": 0.5} for c in ids],
            "latency_ms": 100.0, "tokens": 10,
            "corpus_sha256": ds.corpus_sha256, "queries_sha256": ds.queries_sha256,
        })
    meta = {
        "label": "Fake: model", "dry_run": False, "chunks": len(ds.chunks), "top_k": 5,
        "index_tokens": 1_000_000, "index_seconds_in_embedding_calls": 12.0,
        "index_seconds_waiting_on_rate_limits": 0.0, "index_calls": 4, "dimensions": 8,
        "index_price_per_1m": 0.10, "query_price_per_1m": 0.20,
        "corpus_sha256": ds.corpus_sha256, "queries_sha256": ds.queries_sha256,
    }
    if mutate:
        mutate(lines, meta)
    (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (tmp_path / f"{name}.meta.json").write_text(json.dumps(meta))


@pytest.fixture
def ds(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RAW", tmp_path)
    return load_dataset()


def test_summary_numbers(ds, tmp_path):
    # q01 has 3 labelled chunks and we return one of them at rank 1.
    fake_run(ds, tmp_path)
    row = report.summarize("fake", *report.load_run("fake", ds), ds, {"price_source": "test"})
    assert row["correct"] == 1 and row["misses"] == 29
    assert row["recall"] == pytest.approx((1 / 3) / 30)
    assert row["mrr"] == pytest.approx(1 / 30)
    assert row["index_cost"] == pytest.approx(0.10)
    assert row["query_cost"] == pytest.approx(300 / 1e6 * 0.20)
    assert row["per_1000_correct"] == pytest.approx(row["query_cost"] * 1000)
    assert row["p50_ms"] == 100.0


def test_near_duplicate_misses_are_counted(ds, tmp_path):
    near = ds.near_duplicates["q03"][0]["chunk_id"]
    fake_run(ds, tmp_path, hit_ids=(), top={"q03": [near]})
    row = report.summarize("fake", *report.load_run("fake", ds), ds, {"price_source": "test"})
    assert row["correct"] == 0 and row["near_dup_misses"] == 1


def test_missing_query_is_rejected(ds, tmp_path):
    fake_run(ds, tmp_path, mutate=lambda lines, meta: lines.pop())
    with pytest.raises(report.ResultsMismatchError, match="queries"):
        report.load_run("fake", ds)


def test_different_data_is_rejected(ds, tmp_path):
    fake_run(ds, tmp_path, mutate=lambda lines, meta: meta.update(corpus_sha256="deadbeef"))
    with pytest.raises(report.ResultsMismatchError, match="different data"):
        report.load_run("fake", ds)


def test_dry_run_results_are_rejected(ds, tmp_path):
    fake_run(ds, tmp_path, mutate=lambda lines, meta: meta.update(dry_run=True))
    with pytest.raises(report.ResultsMismatchError, match="full run"):
        report.load_run("fake", ds)


def test_summary_refuses_unfinished_runs_unless_partial(ds, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "SUMMARY", tmp_path / "summary.md")
    monkeypatch.setattr(report, "CHART", tmp_path / "chart.png")
    monkeypatch.setattr(report, "CACHE_DIR", tmp_path)
    cfg = report.load_config()
    names = list(cfg["runs"])
    for name in names[:-2]:
        fake_run(ds, tmp_path, name)
    (tmp_path / f"{names[-1]}.running").write_text("{}")
    with pytest.raises(SystemExit, match="unfinished"):
        report.main([])
    report.main(["--partial"])
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "**Incomplete.**" in text
    for name, status in ((names[-1], "*running*"), (names[-2], "*not run*")):
        label = cfg["runs"][name]["label"]
        row = next(line for line in text.splitlines() if line.startswith(f"| {label} |"))
        cells = [c.strip() for c in row.split("|")[3:-1]]  # everything after setup and price
        assert cells[0] == status and all(c == "" for c in cells[1:])  # no numbers, no guesses


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    values = [1.0] * 20 + [0.0] * 10
    lo, hi = report.bootstrap_ci(values)
    assert (lo, hi) == report.bootstrap_ci(values)  # fixed seed
    assert lo < sum(values) / len(values) < hi
    assert report.bootstrap_ci([0.5] * 30) == (0.5, 0.5)  # no variation, no width
    assert 0.45 < lo < 0.52 and 0.82 < hi < 0.88  # sanity for p = 2/3, n = 30
