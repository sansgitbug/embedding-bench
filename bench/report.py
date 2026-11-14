"""Build results/summary.md and the chart from results/raw/. Measured numbers only.

    python -m bench.report             # refuses unless every configured run has finished
    python -m bench.report --partial   # marks unfinished runs "running" / "not run", no numbers
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from bench import cost, metrics
from bench.data import ROOT, load_dataset
from bench.run import CACHE_DIR, load_config

RAW = ROOT / "results" / "raw"
SUMMARY = ROOT / "results" / "summary.md"
CHART = ROOT / "results" / "accuracy_vs_cost.png"


BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20261114


def bootstrap_ci(values, *, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = BOOTSTRAP_SEED,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-query scores.

    Resamples queries with replacement. The fixed seed means every setup is
    scored on the same 10,000 resamples of the same 30 queries.
    """
    values = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(values), size=(iterations, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


class ResultsMismatchError(RuntimeError):
    """A raw result file was not produced from the same chunks and queries."""


def load_run(name: str, ds) -> tuple[list[dict], dict] | None:
    path = RAW / f"{name}.jsonl"
    if not path.exists():
        return None
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = json.loads((RAW / f"{name}.meta.json").read_text(encoding="utf-8"))
    expected = [(q["query_id"], q["query"], q["relevant_chunk_ids"]) for q in ds.queries]
    got = [(line["query_id"], line["query"], line["relevant_chunk_ids"]) for line in lines]
    if got != expected:
        raise ResultsMismatchError(f"{name}: queries in the raw results differ from data/queries.json")
    hashes = {(line["corpus_sha256"], line["queries_sha256"]) for line in lines}
    hashes.add((meta["corpus_sha256"], meta["queries_sha256"]))
    if hashes != {(ds.corpus_sha256, ds.queries_sha256)}:
        raise ResultsMismatchError(f"{name}: results were produced from different data")
    if meta["dry_run"] or meta["chunks"] != len(ds.chunks):
        raise ResultsMismatchError(f"{name}: not a full run over all {len(ds.chunks)} chunks")
    if any(len(line["top_k"]) != meta["top_k"] for line in lines):
        raise ResultsMismatchError(f"{name}: a query has fewer than {meta['top_k']} results")
    return lines, meta


def summarize(name: str, lines: list[dict], meta: dict, ds, run_cfg: dict) -> dict:
    k = meta["top_k"]
    recalls, rrs, correct, near_dup_misses = [], [], 0, 0
    for line in lines:
        ranked = [t["chunk_id"] for t in line["top_k"]]
        relevant = set(line["relevant_chunk_ids"])
        recalls.append(metrics.recall_at_k(ranked, relevant, k))
        rrs.append(metrics.reciprocal_rank_at_k(ranked, relevant, k))
        if metrics.is_correct(ranked, relevant, k):
            correct += 1
        else:
            near = {d["chunk_id"] for d in ds.near_duplicates.get(line["query_id"], [])}
            near_dup_misses += bool(near & set(ranked))
    latencies = [line["latency_ms"] for line in lines]
    query_tokens = sum(line["tokens"] for line in lines)
    index_cost = cost.cost_usd(meta["index_tokens"], meta["index_price_per_1m"])
    query_cost = cost.cost_usd(query_tokens, meta["query_price_per_1m"])
    return {
        "run": name,
        "status": "done",
        "label": meta["label"],
        "index_price": meta["index_price_per_1m"],
        "query_price": meta["query_price_per_1m"],
        "price_source": run_cfg["price_source"],
        "chart_label": run_cfg.get("chart_label", meta["label"]),
        "recall": float(np.mean(recalls)),
        "recall_ci": bootstrap_ci(recalls),
        "mrr": float(np.mean(rrs)),
        "mrr_ci": bootstrap_ci(rrs),
        "correct": correct,
        "queries": len(lines),
        "misses": len(lines) - correct,
        "near_dup_misses": near_dup_misses,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "index_seconds": meta["index_seconds_in_embedding_calls"],
        "index_wait_seconds": meta["index_seconds_waiting_on_rate_limits"],
        "index_calls": meta["index_calls"],
        "index_tokens": meta["index_tokens"],
        "query_tokens": query_tokens,
        "index_cost": index_cost,
        "query_cost": query_cost,
        "total_cost": index_cost + query_cost,
        "per_1000_correct": cost.cost_per_1000_correct(query_cost, correct),
        "dimensions": meta["dimensions"],
        "device": meta.get("device"),
    }


def pending(name: str, run_cfg: dict) -> dict:
    running = (CACHE_DIR / f"{name}.running").exists()
    return {
        "run": name,
        "status": "running" if running else "not run",
        "label": run_cfg["label"],
        "index_price": run_cfg["index_price_per_1m"],
        "query_price": run_cfg["query_price_per_1m"],
        "price_source": run_cfg["price_source"],
    }


def money(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "$0"
    return f"${value:.4f}" if value >= 0.001 else f"${value:.2g}"


def with_ci(value: float, ci: tuple[float, float]) -> str:
    return f"{value:.3f} ({ci[0]:.2f}–{ci[1]:.2f})"


def price(row: dict) -> str:
    if row["index_price"] == row["query_price"]:
        text = "$0" if row["index_price"] == 0 else f"${row['index_price']:.2f}"
    else:
        text = f"${row['index_price']:.2f} index / ${row['query_price']:.2f} query"
    return text + (" (unverified)" if row["price_source"].startswith("UNVERIFIED") else "")


def write_summary(rows: list[dict], cfg: dict, n_queries: int, n_chunks: int) -> None:
    done = [r for r in rows if r["status"] == "done"]
    unfinished = [r for r in rows if r["status"] != "done"]
    head = [
        "Setup", "List price per 1M tokens", "Recall@5 (95% CI)", "MRR@5 (95% CI)", f"Correct (of {n_queries})",
        "Query p50 (ms)", "Query p95 (ms)", "Indexing time (s)", "Total cost at list price",
        "Cost per 1,000 correct", "Misses that were near-duplicates",
    ]
    out = ["# Results", ""]
    if unfinished:
        out += [
            "> **Incomplete.** "
            + ", ".join(f"{r['label']} is {r['status']}" for r in unfinished)
            + ". Its row has no numbers until the run finishes; nothing here is estimated.",
            "",
        ]
    out += [
        f"{n_queries} queries over {n_chunks} chunks, top 5 by exact cosine similarity. "
        f"List prices were checked on {cfg['checked']} (sources below). Costs are measured tokens "
        "times list price, not what was actually paid: the runs used free tiers and trial keys.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|",
    ]
    for r in rows:
        if r["status"] != "done":
            out.append("| " + " | ".join([r["label"], price(r), f"*{r['status']}*"] + [""] * (len(head) - 3)) + " |")
            continue
        out.append("| " + " | ".join([
            r["label"], price(r), with_ci(r["recall"], r["recall_ci"]), with_ci(r["mrr"], r["mrr_ci"]),
            str(r["correct"]),
            f"{r['p50_ms']:.0f}", f"{r['p95_ms']:.0f}", f"{r['index_seconds']:.0f}",
            money(r["total_cost"]), money(r["per_1000_correct"]),
            f"{r['near_dup_misses']} of {r['misses']}",
        ]) + " |")
    out += [
        "",
        "How to read the columns:",
        "",
        "- **Recall@5** is the share of a query's labelled passages found in the top 5, averaged over queries. "
        "**MRR@5** is 1/rank of the first labelled passage in the top 5 (0 if none), averaged over queries.",
        f"- **95% CI** is a percentile bootstrap over queries: the {n_queries} queries resampled with "
        f"replacement {BOOTSTRAP_ITERATIONS:,} times (seed {BOOTSTRAP_SEED}, the same resamples for every "
        "setup), computed from the saved raw results. Overlapping intervals mean the data cannot separate those setups.",
        "- **Correct** counts queries with at least one labelled passage in the top 5.",
        "- **Query latency** is one embedding call per query, sent one at a time, after one untimed warm-up call. "
        "API latencies include the network round trip from the test machine; local latencies are model time on its CPU.",
        "- **Indexing time** is time spent inside embedding calls for all chunks. It excludes time spent "
        "waiting to stay under rate limits, and batch sizes differ by provider (see the table below).",
        "- **Total cost** is index tokens plus query tokens at list price. "
        "**Cost per 1,000 correct** uses query-embedding cost only: query cost / correct x 1,000.",
        "- **Misses that were near-duplicates**: of the queries scored as misses, how many had a top-5 passage "
        "that is a near-copy (8-word-shingle Jaccard > 0.8) of a labelled passage from a different filing.",
        "",
        "Where each list price came from:",
        "",
    ]
    out += [f"- **{r['label']}**: {r['price_source']}" for r in rows]
    if done:
        out += [
            "",
            "| Setup | Dimensions | Index tokens | Query tokens | Index calls | Rate-limit wait while indexing (s) | Device |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in done:
            out.append(
                f"| {r['label']} | {r['dimensions']} | {r['index_tokens']:,} | {r['query_tokens']:,} | "
                f"{r['index_calls']} | {r['index_wait_seconds']:.0f} | {r['device'] or 'provider API'} |"
            )
        out += ["", "![Recall@5 against cost per 1,000 correct answers](accuracy_vs_cost.png)"]
    out.append("")
    SUMMARY.write_text("\n".join(out), encoding="utf-8", newline="\n")


def write_chart(rows: list[dict], complete: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    groups = {"Local model (free)": "#2a78d6", "Hosted API (list price)": "#eb6834"}
    plotted = [r for r in rows if r["status"] == "done" and r["per_1000_correct"] is not None]

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160, facecolor=surface)
    ax.set_facecolor(surface)
    for group, color in groups.items():
        pts = [r for r in plotted if (r["index_price"] == 0) == group.startswith("Local")]
        if pts:
            ax.scatter([r["per_1000_correct"] for r in pts], [r["recall"] for r in pts], s=90,
                       color=color, edgecolor=surface, linewidth=2, label=group, zorder=3)
    xmax = max([r["per_1000_correct"] for r in plotted] + [1e-6])
    for r in sorted(plotted, key=lambda r: r["recall"]):
        right_side = r["per_1000_correct"] > xmax * 0.6
        ax.annotate(
            r["chart_label"], (r["per_1000_correct"], r["recall"]),
            xytext=(-9 if right_side else 9, 0), textcoords="offset points",
            ha="right" if right_side else "left", va="center", fontsize=8.5, color=ink,
        )
    ax.set_xlim(-xmax * 0.06, xmax * 1.12)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: "$0" if abs(x) < 1e-12 else f"${x:.4f}")
    ax.set_xlabel("Cost per 1,000 correct answers, USD (query embedding at list price)", color=muted, fontsize=9)
    ax.set_ylabel("Recall@5", color=muted, fontsize=9)
    title = f"Retrieval accuracy against cost, {rows[0].get('queries', 30)} SEC-filing queries"
    ax.set_title(title + ("" if complete else " (incomplete)"), loc="left", color=ink, fontsize=12, pad=14)
    ax.grid(color=grid, linewidth=0.8, zorder=0)
    ax.tick_params(colors=muted, labelsize=8.5, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=muted, loc="lower right")
    fig.tight_layout()
    fig.savefig(CHART, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partial", action="store_true", help="write even if some runs have not finished")
    args = parser.parse_args(argv)

    cfg = load_config()
    ds = load_dataset()
    rows = []
    for name, run_cfg in cfg["runs"].items():
        loaded = load_run(name, ds)
        rows.append(summarize(name, *loaded, ds, run_cfg) if loaded else pending(name, run_cfg))
    unfinished = [r["run"] for r in rows if r["status"] != "done"]
    if unfinished and not args.partial:
        raise SystemExit(f"Not writing the summary: unfinished runs: {', '.join(unfinished)}. "
                         "Use --partial to publish an explicitly incomplete table.")
    if len(unfinished) == len(rows):
        raise SystemExit("No finished runs in results/raw/.")
    write_summary(rows, cfg, len(ds.queries), len(ds.chunks))
    write_chart(rows, complete=not unfinished)
    print(f"wrote {SUMMARY} and {CHART}"
          + (f"; incomplete: {', '.join(unfinished)}" if unfinished else ""))


if __name__ == "__main__":
    main()
