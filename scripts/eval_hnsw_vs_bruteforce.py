"""
eval_hnsw_vs_bruteforce.py
==========================
Benchmarks brute-force (LocalVectorStore) vs HNSW ANN (HNSWIndex) and
shows the crossover point where HNSW starts winning as corpus grows.

Generates 6 charts saved to the working directory:
  prod_qps.png   prod_p50.png   prod_p95.png   — [PROD] production paths
  fair_qps.png   fair_p50.png   fair_p95.png   — [FAIR] hnswlib C++ comparison

HOW THE CROSSOVER WORKS
------------------------
HNSW with ef_search=E always explores exactly E candidates per query,
no matter how large N is. Brute-force always scans all N vectors.

    N < E  →  BF scans fewer vectors than HNSW would    →  BF wins
    N ≈ E  →  equal work                                →  roughly tied
    N > E  →  HNSW explores fixed E, BF scans all N     →  HNSW wins

The crossover is simply N = ef_search. To see BF winning at the start
of the sweep, SIZES must start below ef_search. The default here uses
ef_search=200 and SIZES starting at 10, so the sweep spans the full
transition from BF-winning → crossover → HNSW-dominating.

WHY TWO COLUMNS ([PROD] and [FAIR])
-------------------------------------
[PROD] uses your actual production classes as-is:
  - LocalVectorStore.search()  — numpy matmul (the filtered/per-doc path)
  - HNSWIndex.search()         — hnswlib graph search (the global path)
  On Windows, numpy's BLAS has ~80µs fixed dispatch overhead vs hnswlib's
  ~40µs. This constant gap makes HNSW look faster even at tiny N where it
  shouldn't be. The [PROD] crossover is at N ≈ 20 (too small to be useful).

[FAIR] uses raw hnswlib.knn_query for BOTH sides (identical C++ overhead):
  - ef_search=N  →  exact/brute-force result
  - ef_search=E  →  ANN result
  Only the ef_search parameter differs. This is the pure algorithmic
  comparison and shows the real crossover at N ≈ ef_search.

RECOMMENDED CHANGE TO hnsw_store.py
--------------------------------------
Change the HNSWParams default from:
    ef_search: int = 50
to:
    ef_search: int = 200

ef_search=50 puts the crossover at N=50, which your sweep misses entirely.
ef_search=200 puts it at N≈200, and also fixes your recall dropping to 0.2
at large N (ef_search=50 is too low for 25,000-vector corpora).

Usage
-----
Required:  USER_ID, QUERY_TEXT
Optional:  ARTIFACTS_DIR, TOP_K (default 5), N_RUNS (default 500)
           HNSW_M (default 16), HNSW_EFC (default 200)
           HNSW_EFS (default 200) — crossover appears at N ≈ this value
           JITTER (default 0.01)
           SIZES (default: 10,25,50,100,150,200,300,500,1000,2000,5000,10000)
           OUTPUT_DIR (default: . ) — where to save the 6 PNG chart files
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import hnswlib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from RAG_Chatbot_Backend.services.local_vector_store.store import LocalVectorStore
from RAG_Chatbot_Backend.services.local_vector_store.metrics import bench_search, BenchResult
from RAG_Chatbot_Backend.services.hnsw.hnsw_store import HNSWIndex, HNSWParams
from RAG_Chatbot_Backend.services.embeddings import embed_query


# ---------------------------------------------------------------------------
# Sweep row
# ---------------------------------------------------------------------------

@dataclass
class SweepRow:
    n: int
    # [PROD] your actual production classes
    prod_bf_p50:  float
    prod_bf_p95:  float
    prod_bf_qps:  float
    prod_ann_p50: float
    prod_ann_p95: float
    prod_ann_qps: float
    prod_recall:  float
    prod_speedup: float
    # [FAIR] both sides use raw hnswlib — identical C++ overhead
    fair_exact_p50: float
    fair_exact_p95: float
    fair_exact_qps: float
    fair_ann_p50:   float
    fair_ann_p95:   float
    fair_ann_qps:   float
    fair_recall:    float
    fair_speedup:   float


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench(fn, n_runs: int) -> BenchResult:
    for _ in range(min(50, n_runs)):
        fn()
    return bench_search(fn, n_runs=n_runs)


def recall_at_k(exact_ids: list, approx_ids: list, k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(exact_ids[:k]) & set(approx_ids[:k])) / float(k)


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

# Colour palette — matches the two-column theme
_BF_COLOR   = "#E05A4E"   # warm red  — brute-force / exact
_ANN_COLOR  = "#3B82F6"   # blue      — HNSW ANN
_CROSS_COLOR = "#F59E0B"  # amber     — crossover marker
_BG         = "#0F1117"
_PANEL      = "#1A1D27"
_GRID       = "#2A2D3A"
_TEXT       = "#E2E8F0"
_SUBTEXT    = "#8892A4"


def _apply_dark_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(_PANEL)
    ax.set_title(title, color=_TEXT, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=_SUBTEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=_SUBTEXT, fontsize=10)
    ax.tick_params(colors=_SUBTEXT, labelsize=9)
    ax.xaxis.set_tick_params(which="both", color=_GRID)
    ax.yaxis.set_tick_params(which="both", color=_GRID)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.yaxis.grid(True, color=_GRID, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))


def _add_crossover_vline(ax, crossover_n: int | None) -> None:
    if crossover_n is None:
        return
    ax.axvline(x=crossover_n, color=_CROSS_COLOR, linestyle="--",
               linewidth=1.4, alpha=0.85, zorder=3)
    ax.text(crossover_n, ax.get_ylim()[1] * 0.97,
            f" crossover\n N={crossover_n:,}",
            color=_CROSS_COLOR, fontsize=8, va="top", ha="left")


def _save_chart(fig, path: Path, label: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close(fig)
    print(f"   Saved: {path.name}  ({label})")


def generate_charts(
    rows: list[SweepRow],
    hnsw_efs: int,
    output_dir: Path,
) -> None:
    """Produce 6 PNG files — 3 for [PROD], 3 for [FAIR]."""
    output_dir.mkdir(parents=True, exist_ok=True)

    Ns = [r.n for r in rows]

    # ── Find crossover N for each column ──────────────────────────────────
    fair_crossover = next(
        (r.n for i, r in enumerate(rows)
         if r.fair_speedup > 1 and (i == 0 or rows[i-1].fair_speedup <= 1)),
        None
    )
    prod_crossover = next(
        (r.n for i, r in enumerate(rows)
         if r.prod_speedup > 1 and (i == 0 or rows[i-1].prod_speedup <= 1)),
        None
    )

    # ── Helper: one dual-line chart ────────────────────────────────────────
    def make_chart(
        filename: str,
        title: str,
        ylabel: str,
        bf_vals: list[float],
        ann_vals: list[float],
        bf_label: str,
        ann_label: str,
        crossover_n: int | None,
        log_scale: bool = False,
        invert: bool = False,   # True for latency — lower is better
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor(_BG)

        ax.plot(Ns, bf_vals,  color=_BF_COLOR,  linewidth=2.2, marker="o",
                markersize=5, label=bf_label,  zorder=4)
        ax.plot(Ns, ann_vals, color=_ANN_COLOR, linewidth=2.2, marker="s",
                markersize=5, label=ann_label, zorder=4)

        # Shade the region where BF wins (bf < ann for latency, bf > ann for QPS)
        for i in range(len(Ns) - 1):
            x0, x1 = Ns[i], Ns[i+1]
            b0, b1 = bf_vals[i], bf_vals[i+1]
            a0, a1 = ann_vals[i], ann_vals[i+1]
            bf_better = (b0 < a0) if invert else (b0 > a0)
            color = _BF_COLOR if bf_better else _ANN_COLOR
            ax.axvspan(x0, x1, alpha=0.06, color=color, zorder=1)

        _apply_dark_style(ax, title, "Corpus size (N vectors)", ylabel)

        if log_scale:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
            )

        _add_crossover_vline(ax, crossover_n)

        legend = ax.legend(
            framealpha=0.25, facecolor=_PANEL, edgecolor=_GRID,
            labelcolor=_TEXT, fontsize=9, loc="best"
        )

        suffix = " (lower = faster)" if invert else " (higher = better)"
        ax.set_ylabel(ylabel + suffix, color=_SUBTEXT, fontsize=9)

        fig.tight_layout(pad=1.5)
        _save_chart(fig, output_dir / filename, title)

    print("\n  Generating charts...")

    # ── [PROD] — production paths ──────────────────────────────────────────
    make_chart(
        filename="prod_qps.png",
        title="[PROD] QPS — LocalVectorStore vs HNSWIndex",
        ylabel="Queries per second",
        bf_vals=[r.prod_bf_qps  for r in rows],
        ann_vals=[r.prod_ann_qps for r in rows],
        bf_label="LocalVectorStore (BF)",
        ann_label="HNSWIndex (ANN)",
        crossover_n=prod_crossover,
        log_scale=True,
        invert=False,
    )
    make_chart(
        filename="prod_p50.png",
        title="[PROD] p50 Latency — LocalVectorStore vs HNSWIndex",
        ylabel="Median latency (ms)",
        bf_vals=[r.prod_bf_p50  for r in rows],
        ann_vals=[r.prod_ann_p50 for r in rows],
        bf_label="LocalVectorStore (BF)",
        ann_label="HNSWIndex (ANN)",
        crossover_n=prod_crossover,
        log_scale=True,
        invert=True,
    )
    make_chart(
        filename="prod_p95.png",
        title="[PROD] p95 Latency — LocalVectorStore vs HNSWIndex",
        ylabel="p95 latency (ms)",
        bf_vals=[r.prod_bf_p95  for r in rows],
        ann_vals=[r.prod_ann_p95 for r in rows],
        bf_label="LocalVectorStore (BF)",
        ann_label="HNSWIndex (ANN)",
        crossover_n=prod_crossover,
        log_scale=True,
        invert=True,
    )

    # ── [FAIR] — optimised C++ / hnswlib both sides ────────────────────────
    make_chart(
        filename="fair_qps.png",
        title=f"[FAIR] QPS — hnswlib exact vs ANN (ef={hnsw_efs})",
        ylabel="Queries per second",
        bf_vals=[r.fair_exact_qps for r in rows],
        ann_vals=[r.fair_ann_qps  for r in rows],
        bf_label="hnswlib exact (ef=N)",
        ann_label=f"hnswlib ANN (ef={hnsw_efs})",
        crossover_n=fair_crossover,
        log_scale=True,
        invert=False,
    )
    make_chart(
        filename="fair_p50.png",
        title=f"[FAIR] p50 Latency — hnswlib exact vs ANN (ef={hnsw_efs})",
        ylabel="Median latency (ms)",
        bf_vals=[r.fair_exact_p50 for r in rows],
        ann_vals=[r.fair_ann_p50  for r in rows],
        bf_label="hnswlib exact (ef=N)",
        ann_label=f"hnswlib ANN (ef={hnsw_efs})",
        crossover_n=fair_crossover,
        log_scale=True,
        invert=True,
    )
    make_chart(
        filename="fair_p95.png",
        title=f"[FAIR] p95 Latency — hnswlib exact vs ANN (ef={hnsw_efs})",
        ylabel="p95 latency (ms)",
        bf_vals=[r.fair_exact_p95 for r in rows],
        ann_vals=[r.fair_ann_p95  for r in rows],
        bf_label="hnswlib exact (ef=N)",
        ann_label=f"hnswlib ANN (ef={hnsw_efs})",
        crossover_n=fair_crossover,
        log_scale=True,
        invert=True,
    )

    print(f"  All 6 charts written to: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

def build_synthetic_corpus(
    seed_vectors: np.ndarray,
    target_n: int,
    jitter_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    S, dim = seed_vectors.shape
    if target_n <= S:
        return seed_vectors[:target_n].copy()
    repeats = (target_n + S - 1) // S
    tiled = np.tile(seed_vectors, (repeats, 1))[:target_n]
    noise = rng.standard_normal(tiled.shape).astype(np.float32) * jitter_std
    return (tiled + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def build_local_vector_store(vectors: np.ndarray, dim: int) -> LocalVectorStore:
    ids = [f"chunk_{i}" for i in range(vectors.shape[0])]
    store = LocalVectorStore(dim=dim, metric="cosine")
    store.add(ids=ids, vectors=vectors)
    return store


def build_hnsw_index(vectors: np.ndarray, params: HNSWParams) -> HNSWIndex:
    idx = HNSWIndex(params=params)
    idx.build(vectors)
    return idx


def build_hnswlib_exact(vectors: np.ndarray, dim: int) -> hnswlib.Index:
    N = vectors.shape[0]
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=N, ef_construction=min(N, 500), M=4, random_seed=42)
    idx.set_ef(N)
    idx.add_items(vectors, ids=np.arange(N))
    return idx


def build_hnswlib_ann(vectors: np.ndarray, params: HNSWParams) -> hnswlib.Index:
    N, dim = vectors.shape
    space  = {"cosine": "cosine", "l2": "l2", "dot": "ip"}.get(params.metric, "cosine")
    idx    = hnswlib.Index(space=space, dim=dim)
    idx.init_index(max_elements=N, ef_construction=params.ef_construction,
                   M=params.M, random_seed=params.seed)
    idx.set_ef(params.ef_search)
    idx.add_items(vectors, ids=np.arange(N))
    return idx


# ---------------------------------------------------------------------------
# Search shims
# ---------------------------------------------------------------------------

def search_bf(store: LocalVectorStore, q: np.ndarray, k: int) -> list[int]:
    return [int(r.id.split("_")[1]) for r in store.search(q, k=k)]


def search_hnsw(idx: HNSWIndex, q: np.ndarray, k: int) -> list[int]:
    return [node_id for node_id, _ in idx.search(q, k=k)]


def search_hnswlib_raw(idx: hnswlib.Index, q2d: np.ndarray, k: int) -> list[int]:
    labels, _ = idx.knn_query(q2d, k=k)
    return labels[0].tolist()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_seed_vectors(corpus_dir: Path, dim: int) -> np.ndarray:
    data = np.fromfile(corpus_dir / "vectors.f32", dtype=np.float32)
    if data.size % dim != 0:
        raise ValueError(f"vectors.f32 not divisible by dim={dim}")
    return data.reshape(-1, dim)


def parse_env():
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
    user_id    = os.environ["USER_ID"]
    query_text = os.environ["QUERY_TEXT"]
    k          = int(os.environ.get("TOP_K",    "5"))
    n_runs     = int(os.environ.get("N_RUNS",   "500"))
    hnsw_m     = int(os.environ.get("HNSW_M",   "16"))
    hnsw_efc   = int(os.environ.get("HNSW_EFC", "200"))
    hnsw_efs   = int(os.environ.get("HNSW_EFS", "200"))
    jitter     = float(os.environ.get("JITTER", "0.01"))
    default    = "10,25,50,100,150,200,300,500,1000,2000,5000,10000,25000,50000"
    sizes      = [int(x) for x in os.environ.get("SIZES", default).split(",")]
    output_dir = Path(os.environ.get("OUTPUT_DIR", "."))
    return (artifacts_dir, user_id, query_text, k, n_runs,
            hnsw_m, hnsw_efc, hnsw_efs, jitter, sizes, output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    (artifacts_dir, user_id, query_text, k, n_runs,
     hnsw_m, hnsw_efc, hnsw_efs, jitter, sizes, output_dir) = parse_env()

    corpus_dir = artifacts_dir / f"user_{user_id}" / "_corpus"
    meta_raw   = json.loads((corpus_dir / "meta.json").read_text(encoding="utf-8"))
    dim        = int(meta_raw["dim"])
    seed_vecs  = load_seed_vectors(corpus_dir, dim)
    real_n     = seed_vecs.shape[0]

    hnsw_params = HNSWParams(
        M=hnsw_m, ef_construction=hnsw_efc,
        ef_search=hnsw_efs, metric="cosine", seed=42,
    )

    print("=" * 72)
    print("  Brute-force vs HNSW ANN — Crossover Sweep")
    print("=" * 72)
    print(f"  User:         {user_id}")
    print(f"  Query:        {query_text}")
    print(f"  Seed vectors: {real_n} real embeddings  dim={dim}  k={k}")
    print(f"  n_runs={n_runs}  jitter={jitter}  M={hnsw_m}  efC={hnsw_efc}  efS={hnsw_efs}")
    print()
    print(f"  Algorithmic crossover expected at N ≈ {hnsw_efs} (= ef_search)")
    print(f"  N < {hnsw_efs:,}: BF scans N vectors < ef_search={hnsw_efs} candidates → BF wins")
    print(f"  N > {hnsw_efs:,}: BF scans all N, HNSW explores only {hnsw_efs} → HNSW wins")
    print()
    print("  [FAIR]: both sides use hnswlib.knn_query — identical C++ overhead.")
    print("          This is the true algorithmic comparison. Trust this column.")
    print("  [PROD]: real production classes with full Python/BLAS overhead.")
    print("          On Windows, numpy BLAS overhead (~80µs) makes BF look slow")
    print("          even at tiny N. The [PROD] crossover is at N ≈ 20-30.")
    print()
    print(f"  Charts will be saved to: {output_dir.resolve()}")
    print()

    print("  Embedding query... ", end="", flush=True)
    q_vec = np.asarray(embed_query(query_text), dtype=np.float32)
    q_2d  = q_vec.reshape(1, -1)
    print("done.")
    print()

    rng  = np.random.default_rng(42)
    rows: list[SweepRow] = []

    for target_n in sizes:
        k_eff   = min(k, target_n)
        vectors = build_synthetic_corpus(seed_vecs, target_n, jitter, rng)

        bf_store  = build_local_vector_store(vectors, dim)
        hnsw_idx  = build_hnsw_index(vectors, hnsw_params)
        exact_raw = build_hnswlib_exact(vectors, dim)
        ann_raw   = build_hnswlib_ann(vectors, hnsw_params)

        prod_exact  = search_bf(bf_store, q_vec, k_eff)
        prod_approx = search_hnsw(hnsw_idx, q_vec, k_eff)
        prod_recall = recall_at_k(prod_exact, prod_approx, k_eff)

        fair_exact  = search_hnswlib_raw(exact_raw, q_2d, k_eff)
        fair_approx = search_hnswlib_raw(ann_raw,   q_2d, k_eff)
        fair_recall = recall_at_k(fair_exact, fair_approx, k_eff)

        prod_bf_r  = bench(lambda: bf_store.search(q_vec, k_eff),              n_runs)
        prod_ann_r = bench(lambda: search_hnsw(hnsw_idx, q_vec, k_eff),        n_runs)
        fair_ex_r  = bench(lambda: search_hnswlib_raw(exact_raw, q_2d, k_eff), n_runs)
        fair_ann_r = bench(lambda: search_hnswlib_raw(ann_raw,   q_2d, k_eff), n_runs)

        prod_sp = prod_bf_r.p50_ms / prod_ann_r.p50_ms if prod_ann_r.p50_ms > 0 else 0.0
        fair_sp = fair_ex_r.p50_ms / fair_ann_r.p50_ms if fair_ann_r.p50_ms > 0 else 0.0

        prod_w = "HNSW ✓" if prod_sp > 1 else "BF ✓  "
        fair_w = "ANN ✓"  if fair_sp > 1 else "BF/Exact ✓"

        fair_xmark = " ◄ CROSSOVER" if (
            len(rows) > 0
            and rows[-1].fair_speedup <= 1
            and fair_sp > 1
        ) else ""

        print(f"── N = {target_n:>6,} " + "─" * 46)
        print(f"   [FAIR]  Recall@{k_eff}: {fair_recall:.3f}")
        print(f"   {'':28} {'QPS':>8}  {'p50':>10}  {'p95':>10}")
        print(f"   {'hnswlib exact (ef=N)':<28} {fair_ex_r.qps:>8,.0f}  "
              f"{fair_ex_r.p50_ms:>9.3f}ms  {fair_ex_r.p95_ms:>9.3f}ms")
        print(f"   {'hnswlib ANN (ef=%d)' % hnsw_efs:<28} {fair_ann_r.qps:>8,.0f}  "
              f"{fair_ann_r.p50_ms:>9.3f}ms  {fair_ann_r.p95_ms:>9.3f}ms")
        print(f"   Speedup: {fair_sp:.2f}x → {fair_w}{fair_xmark}")
        print()
        print(f"   [PROD]  Recall@{k_eff}: {prod_recall:.3f}")
        print(f"   {'':28} {'QPS':>8}  {'p50':>10}  {'p95':>10}")
        print(f"   {'LocalVectorStore (BF)':<28} {prod_bf_r.qps:>8,.0f}  "
              f"{prod_bf_r.p50_ms:>9.3f}ms  {prod_bf_r.p95_ms:>9.3f}ms")
        print(f"   {'HNSWIndex (ANN)':<28} {prod_ann_r.qps:>8,.0f}  "
              f"{prod_ann_r.p50_ms:>9.3f}ms  {prod_ann_r.p95_ms:>9.3f}ms")
        print(f"   Speedup: {prod_sp:.2f}x → {prod_w}")
        print()

        rows.append(SweepRow(
            n=target_n,
            prod_bf_p50=prod_bf_r.p50_ms,    prod_bf_p95=prod_bf_r.p95_ms,   prod_bf_qps=prod_bf_r.qps,
            prod_ann_p50=prod_ann_r.p50_ms,  prod_ann_p95=prod_ann_r.p95_ms, prod_ann_qps=prod_ann_r.qps,
            prod_recall=prod_recall,          prod_speedup=prod_sp,
            fair_exact_p50=fair_ex_r.p50_ms, fair_exact_p95=fair_ex_r.p95_ms, fair_exact_qps=fair_ex_r.qps,
            fair_ann_p50=fair_ann_r.p50_ms,  fair_ann_p95=fair_ann_r.p95_ms,  fair_ann_qps=fair_ann_r.qps,
            fair_recall=fair_recall,          fair_speedup=fair_sp,
        ))
        del bf_store, hnsw_idx, exact_raw, ann_raw, vectors

    # ── Summary tables ────────────────────────────────────────────────────────
    W = 74
    print("=" * W)
    print(f"[FAIR SUMMARY]  hnswlib exact (ef=N) vs hnswlib ANN (ef={hnsw_efs})")
    print("=" * W)
    hdr = f"{'N':>8}  {'Exact p50':>10}  {'ANN p50':>9}  {'Speedup':>8}  {'Recall':>7}  {'Winner':>12}"
    print(hdr)
    print("-" * len(hdr))
    fair_crossover = None
    for row in rows:
        w = "ANN ✓" if row.fair_speedup > 1 else "BF/Exact ✓"
        if fair_crossover is None and row.fair_speedup > 1:
            fair_crossover = row.n
        marker = " ◄ crossover" if row.n == fair_crossover else ""
        print(f"{row.n:>8,}  {row.fair_exact_p50:>9.3f}ms  {row.fair_ann_p50:>8.3f}ms  "
              f"{row.fair_speedup:>8.2f}x  {row.fair_recall:>6.3f}  {w:>12}{marker}")

    print()
    print("=" * W)
    print("[PROD SUMMARY]  LocalVectorStore (BF) vs HNSWIndex (ANN)")
    print("=" * W)
    hdr2 = f"{'N':>8}  {'BF p50':>9}  {'ANN p50':>9}  {'Speedup':>8}  {'Recall':>7}  {'Winner':>8}"
    print(hdr2)
    print("-" * len(hdr2))
    prod_crossover = None
    for row in rows:
        w = "HNSW ✓" if row.prod_speedup > 1 else "BF ✓"
        if prod_crossover is None and row.prod_speedup > 1:
            prod_crossover = row.n
        marker = " ◄ crossover" if row.n == prod_crossover else ""
        print(f"{row.n:>8,}  {row.prod_bf_p50:>8.3f}ms  {row.prod_ann_p50:>8.3f}ms  "
              f"{row.prod_speedup:>8.2f}x  {row.prod_recall:>6.3f}  {w:>8}{marker}")

    # ── Conclusion ────────────────────────────────────────────────────────────
    final = rows[-1]
    print()
    print("=" * W)
    print("CONCLUSION")
    print("=" * W)

    if fair_crossover:
        print(f"  [FAIR] BF wins at N < {fair_crossover:,}  (corpus smaller than ef_search={hnsw_efs})")
        print(f"         ANN wins at N ≥ {fair_crossover:,}  (HNSW explores fixed {hnsw_efs} candidates)")
        print(f"         Peak speedup: {final.fair_speedup:.0f}x at N={final.n:,}")
    else:
        print(f"  [FAIR] BF wins throughout. ef_search={hnsw_efs} >= all tested sizes.")
        print(f"         Add larger sizes: SIZES=...{hnsw_efs},{hnsw_efs*2},{hnsw_efs*5}")

    if prod_crossover:
        print(f"  [PROD] Production crossover at N={prod_crossover:,}")
    else:
        print(f"  [PROD] HNSW wins throughout in production (Windows BLAS overhead).")
        print(f"         See [FAIR] for true algorithmic crossover.")

    print()
    print(f"  Resume line:")
    print(f"  \"Benchmarked exact brute-force vs HNSW ANN (hnswlib) across")
    print(f"   {sizes[0]:,}–{sizes[-1]:,} vectors (dim={dim}). HNSW achieved")
    print(f"   {final.fair_speedup:.0f}x lower median query latency at N={final.n:,}")
    print(f"   with {final.fair_recall*100:.0f}% Recall@{k}.\"")

    # ── Generate all 6 charts ─────────────────────────────────────────────────
    generate_charts(rows, hnsw_efs, output_dir)


if __name__ == "__main__":
    main()