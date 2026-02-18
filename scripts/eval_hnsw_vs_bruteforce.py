import os
import time
import json
from pathlib import Path

import numpy as np

from RAG_Chatbot_Backend.services.local_vector_store.store import LocalVectorStore
from RAG_Chatbot_Backend.services.hnsw.hnsw_persist import load_hnsw
from RAG_Chatbot_Backend.services.local_vector_store.metrics import bench_search


def recall_at_k(exact_ids, approx_ids, k: int) -> float:
    exact = set(exact_ids[:k])
    approx = set(approx_ids[:k])
    return len(exact.intersection(approx)) / float(k)


def main():
    leaf = Path(os.environ["ARTIFACT_LEAF"])
    dim = int(os.environ.get("EMBED_DIM", "384"))
    k = int(os.environ.get("TOP_K", "5"))
    n_queries = int(os.environ.get("N_QUERIES", "50"))

    # Load vectors
    vectors = np.fromfile(leaf / "embeddings.bin", dtype=np.float32).reshape(-1, dim)

    # Load docstore ids for mapping (optional)
    docstore = []
    with (leaf / "docstore.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            docstore.append(json.loads(line))
    ids = [row["chunk_id"] for row in docstore]

    # Build brute-force baseline store
    bf = LocalVectorStore(dim=dim, metric="cosine")
    bf.add(ids=ids, vectors=vectors, metas=docstore, compute_norms=True)

    # Load HNSW index graph
    hnsw = load_hnsw(leaf / "_hnsw_index")
    # IMPORTANT: attach vectors to loaded index (graph doesn’t store vectors)
    hnsw.vectors = hnsw._prepare_vectors(vectors)
    hnsw.N, hnsw.dim = hnsw.vectors.shape

    # Select query vectors (use stored vectors as realistic embeddings)
    rng = np.random.default_rng(123)
    q_idx = rng.integers(0, len(vectors), size=min(n_queries, len(vectors)))
    queries = vectors[q_idx]

    # Accuracy: recall@k against brute-force
    recalls = []
    for q in queries:
        exact = bf.search(q, k=k)
        exact_ids = [r.id for r in exact]

        approx = hnsw.search(q, k=k)
        approx_ids = [ids[i] for i, _score in approx]

        recalls.append(recall_at_k(exact_ids, approx_ids, k=k))

    avg_recall = float(np.mean(recalls))

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

    bf_bench = bench_search(run_bf, n_runs=300)
    hnsw_bench = bench_search(run_hnsw, n_runs=300)

    print("=== HNSW vs Brute-force ===")
    print(f"Vectors: {len(vectors)} | dim: {dim} | k: {k}")
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
