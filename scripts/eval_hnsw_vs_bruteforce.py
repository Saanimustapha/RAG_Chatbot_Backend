import os
import json
from pathlib import Path

import numpy as np

from RAG_Chatbot_Backend.services.local_vector_store.store import LocalVectorStore
from RAG_Chatbot_Backend.services.local_vector_store.metrics import bench_search
from RAG_Chatbot_Backend.services.hnsw.hnsw_persist import load_hnsw


def recall_at_k(exact_ids, approx_ids, k: int) -> float:
    exact = set(exact_ids[:k])
    approx = set(approx_ids[:k])
    return len(exact.intersection(approx)) / float(k) if k > 0 else 0.0


def load_meta(corpus_dir: Path) -> dict:
    return json.loads((corpus_dir / "meta.json").read_text(encoding="utf-8"))


def load_ids(corpus_dir: Path) -> list[str]:
    return (corpus_dir / "ids.txt").read_text(encoding="utf-8").splitlines()


def load_deleted_ids(corpus_dir: Path) -> set[str]:
    p = corpus_dir / "deleted_ids.txt"
    if not p.exists():
        return set()
    txt = p.read_text(encoding="utf-8").strip()
    return set(txt.splitlines()) if txt else set()


def load_docstore(corpus_dir: Path) -> list[dict]:
    rows = []
    with (corpus_dir / "docstore.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_vectors(corpus_dir: Path, dim: int) -> np.ndarray:
    data = np.fromfile(corpus_dir / "vectors.f32", dtype=np.float32)
    if data.size % dim != 0:
        raise ValueError(f"vectors.f32 size {data.size} not divisible by dim={dim}")
    return data.reshape(-1, dim)


def main():
    """
    Env vars expected:

    ARTIFACTS_DIR   e.g. "artifacts"
    USER_ID         e.g. "2d6e...uuid..."
    TOP_K           default 5
    N_QUERIES        default 50
    N_RUNS           default 300   (for latency benchmark)
    """

    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
    user_id = os.environ["USER_ID"]

    k = int(os.environ.get("TOP_K", "5"))
    n_queries = int(os.environ.get("N_QUERIES", "50"))
    n_runs = int(os.environ.get("N_RUNS", "300"))

    corpus_dir = artifacts_dir / f"user_{user_id}" / "_corpus"
    hnsw_dir = corpus_dir / "_hnsw"

    if not (corpus_dir / "vectors.f32").exists():
        raise FileNotFoundError(f"Missing {corpus_dir / 'vectors.f32'}")
    if not (corpus_dir / "meta.json").exists():
        raise FileNotFoundError(f"Missing {corpus_dir / 'meta.json'}")
    if not (hnsw_dir / "hnsw_meta.json").exists():
        raise FileNotFoundError(f"Missing HNSW index at {hnsw_dir / 'hnsw_meta.json'}")

    # Load corpus data
    meta = load_meta(corpus_dir)
    dim = int(meta["dim"])
    ids = load_ids(corpus_dir)
    deleted = load_deleted_ids(corpus_dir)
    docstore = load_docstore(corpus_dir)
    vectors = load_vectors(corpus_dir, dim=dim)

    if len(ids) != vectors.shape[0]:
        raise ValueError(f"ids.txt count ({len(ids)}) != vectors rows ({vectors.shape[0]})")

    # Build brute-force baseline store
    bf = LocalVectorStore(dim=dim, metric="cosine")
    # metas length can be less than ids if something is off; guard it
    metas = docstore if len(docstore) == len(ids) else [{} for _ in ids]
    bf.add(ids=ids, vectors=vectors, metas=metas, compute_norms=True)

    # Load HNSW index graph and attach vectors (graph doesn’t store vectors)
    hnsw = load_hnsw(hnsw_dir)
    hnsw.vectors = hnsw._prepare_vectors(vectors)
    hnsw.N, hnsw.dim = hnsw.vectors.shape

    # Choose realistic query vectors (sample from existing vectors)
    rng = np.random.default_rng(123)
    q_idx = rng.integers(0, len(vectors), size=min(n_queries, len(vectors)))
    queries = vectors[q_idx]

    # Helper: filter out deleted IDs from result lists (fair comparison)
    def filter_deleted_id_list(id_list: list[str]) -> list[str]:
        if not deleted:
            return id_list
        return [x for x in id_list if x not in deleted]

    # Accuracy: recall@k against brute-force
    recalls = []
    for q in queries:
        exact = bf.search(q, k=max(k * 5, k))  # overfetch then filter
        exact_ids = filter_deleted_id_list([r.id for r in exact])[:k]

        approx = hnsw.search(q, k=max(k * 20, k))  # overfetch then filter
        approx_ids = filter_deleted_id_list([ids[i] for i, _score in approx])[:k]

        # If filtering leaves us with fewer than k, recall@k is still computed against k
        # (you can also compute recall@len(exact_ids) if you prefer)
        if len(exact_ids) < k or len(approx_ids) < k:
            # pad with empties so recall_at_k behaves consistently
            exact_ids = (exact_ids + [""] * k)[:k]
            approx_ids = (approx_ids + [""] * k)[:k]

        recalls.append(recall_at_k(exact_ids, approx_ids, k=k))

    avg_recall = float(np.mean(recalls)) if recalls else 0.0

    # Latency benchmarks
    qi = 0

    def run_bf():
        nonlocal qi
        bf.search(queries[qi % len(queries)], k=k)
        qi += 1

    qi2 = 0

    def run_hnsw():
        nonlocal qi2
        hnsw.search(queries[qi2 % len(queries)], k=k)
        qi2 += 1

    bf_bench = bench_search(run_bf, n_runs=n_runs)
    hnsw_bench = bench_search(run_hnsw, n_runs=n_runs)

    print("=== HNSW vs Brute-force (User Corpus) ===")
    print(f"User: {user_id}")
    print(f"Vectors: {len(vectors)} | dim: {dim} | k: {k} | queries: {len(queries)}")
    print(f"Deleted IDs: {len(deleted)}")
    print(f"Recall@{k}: {avg_recall:.3f}")
    print()
    print("Brute-force:")
    print(f"  QPS: {bf_bench.qps:.2f}, p50: {bf_bench.p50_ms:.3f} ms, p95: {bf_bench.p95_ms:.3f} ms")
    print("HNSW:")
    print(f"  QPS: {hnsw_bench.qps:.2f}, p50: {hnsw_bench.p50_ms:.3f} ms, p95: {hnsw_bench.p95_ms:.3f} ms")

    speedup_p95 = (bf_bench.p95_ms / hnsw_bench.p95_ms) if hnsw_bench.p95_ms > 0 else 0.0
    print(f"\nSpeedup (p95): {speedup_p95:.2f}x")


if __name__ == "__main__":
    main()