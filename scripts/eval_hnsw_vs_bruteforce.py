import os
import json
from pathlib import Path
from typing import List

import numpy as np

from RAG_Chatbot_Backend.services.local_vector_store.store import LocalVectorStore
from RAG_Chatbot_Backend.services.local_vector_store.metrics import bench_search
from RAG_Chatbot_Backend.services.hnsw.hnsw_store import load_hnsw, save_hnsw, HNSWIndex, HNSWParams

# Uses your actual embedding pipeline for query text
from RAG_Chatbot_Backend.services.embeddings import embed_query


def recall_at_k(exact_ids: List[str], approx_ids: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    exact = set(exact_ids[:k])
    approx = set(approx_ids[:k])
    return len(exact.intersection(approx)) / float(k)



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


def parse_env():
    """
    Required:
      USER_ID
      QUERY_TEXT

    Optional:
      ARTIFACTS_DIR (default: artifacts)
      TOP_K (default: 5)
      N_RUNS (default: 300)

      # If your persisted HNSW is stale, rebuild with these defaults:
      HNSW_M (default: 16)
      HNSW_EFC (default: 200)
      HNSW_EFS (default: 80)
    """
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
    user_id = os.environ["USER_ID"]
    query_text = os.environ["QUERY_TEXT"]

    k = int(os.environ.get("TOP_K", "5"))
    n_runs = int(os.environ.get("N_RUNS", "300"))

    hnsw_m = int(os.environ.get("HNSW_M", "16"))
    hnsw_efc = int(os.environ.get("HNSW_EFC", "200"))
    hnsw_efs = int(os.environ.get("HNSW_EFS", "80"))

    return artifacts_dir, user_id, query_text, k, n_runs, hnsw_m, hnsw_efc, hnsw_efs


def main():
    artifacts_dir, user_id, query_text, k, n_runs, hnsw_m, hnsw_efc, hnsw_efs = parse_env()

    corpus_dir = artifacts_dir / f"user_{user_id}" / "_corpus"
    hnsw_dir = corpus_dir / "_hnsw"

    if not (corpus_dir / "vectors.f32").exists():
        raise FileNotFoundError(f"Missing {corpus_dir / 'vectors.f32'}")
    if not (corpus_dir / "meta.json").exists():
        raise FileNotFoundError(f"Missing {corpus_dir / 'meta.json'}")
    if not (corpus_dir / "ids.txt").exists():
        raise FileNotFoundError(f"Missing {corpus_dir / 'ids.txt'}")
    if not (corpus_dir / "docstore.jsonl").exists():
        raise FileNotFoundError(f"Missing {corpus_dir / 'docstore.jsonl'}")

    # Load corpus
    meta = load_meta(corpus_dir)
    dim = int(meta["dim"])
    ids = load_ids(corpus_dir)
    deleted = load_deleted_ids(corpus_dir)
    docstore = load_docstore(corpus_dir)
    vectors = load_vectors(corpus_dir, dim=dim)

    if len(ids) != vectors.shape[0]:
        raise ValueError(f"ids.txt count ({len(ids)}) != vectors rows ({vectors.shape[0]})")

    # Build brute-force baseline store over ALL vectors (global brute force)
    bf = LocalVectorStore(dim=dim, metric="cosine")
    metas = docstore if len(docstore) == len(ids) else [{} for _ in ids]
    bf.add(ids=ids, vectors=vectors, metas=metas, compute_norms=True)

    # Load GLOBAL HNSW index (hnswlib — C++ speed, no stale-graph risk)
    hnsw = load_hnsw(hnsw_dir)

    if hnsw._index is None or hnsw.N == 0:
        print("[INFO] No valid hnswlib index found. Building from scratch...")
        params = HNSWParams(M=hnsw_m, ef_construction=hnsw_efc, ef_search=hnsw_efs, metric="cosine", seed=42)
        hnsw = HNSWIndex(params=params)
        hnsw.build(vectors)
        save_hnsw(hnsw, hnsw_dir)
        print(f"[OK] Built HNSW index with {hnsw.N} vectors.")

    # Embed the provided query text
    q_emb = embed_query(query_text)
    q = np.asarray(q_emb, dtype=np.float32).reshape(dim,)

    # Helper: remove deleted IDs from lists (fairness)
    def filter_deleted_id_list(id_list: list[str]) -> list[str]:
        if not deleted:
            return id_list
        return [x for x in id_list if x not in deleted]

    # Accuracy: recall@k against global brute force
    exact = bf.search(q, k=max(k * 5, k))  # overfetch then filter deleted
    exact_ids = filter_deleted_id_list([r.id for r in exact])[:k]

    approx = hnsw.search(q, k=max(k * 20, k))  # overfetch then filter deleted
    approx_ids = filter_deleted_id_list([ids[i] for i, _score in approx])[:k]

    # pad if needed
    if len(exact_ids) < k or len(approx_ids) < k:
        exact_ids = (exact_ids + [""] * k)[:k]
        approx_ids = (approx_ids + [""] * k)[:k]

    rec = recall_at_k(exact_ids, approx_ids, k=k)

    # Latency benchmarks (single query repeated)
    def run_bf():
        bf.search(q, k=k)

    def run_hnsw():
        hnsw.search(q, k=k)

    bf_bench = bench_search(run_bf, n_runs=n_runs)
    hnsw_bench = bench_search(run_hnsw, n_runs=n_runs)

    # Summary
    print("=== Brute-force vs HNSW (GLOBAL index, Query-text driven) ===")
    print(f"Artifacts: {artifacts_dir}")
    print(f"User:      {user_id}")
    print(f"Query:     {query_text}")
    print()
    print(f"Vectors: {len(vectors)} | dim: {dim} | k: {k}")
    print(f"Deleted IDs: {len(deleted)}")
    print(f"HNSW params (if rebuilt): M={hnsw_m}, efC={hnsw_efc}, efS={hnsw_efs}")
    print(f"Recall@{k}: {rec:.3f}")
    print()
    print("Brute-force (global exact):")
    print(f"  QPS: {bf_bench.qps:.2f}, p50: {bf_bench.p50_ms:.3f} ms, p95: {bf_bench.p95_ms:.3f} ms")
    print("HNSW (global ANN):")
    print(f"  QPS: {hnsw_bench.qps:.2f}, p50: {hnsw_bench.p50_ms:.3f} ms, p95: {hnsw_bench.p95_ms:.3f} ms")

    speedup_p95 = (bf_bench.p95_ms / hnsw_bench.p95_ms) if hnsw_bench.p95_ms > 0 else 0.0
    print(f"\nSpeedup (p95): {speedup_p95:.2f}x")


if __name__ == "__main__":
    main()