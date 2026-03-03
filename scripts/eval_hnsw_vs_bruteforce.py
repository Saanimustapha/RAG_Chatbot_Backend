import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from RAG_Chatbot_Backend.services.local_vector_store.store import LocalVectorStore
from RAG_Chatbot_Backend.services.local_vector_store.metrics import bench_search
from RAG_Chatbot_Backend.services.hnsw.hnsw_index import HNSWIndex, HNSWParams

# ✅ Uses your actual embedding pipeline for query text
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
      DOCUMENT_ID
      QUERY_TEXT

    Optional:
      ARTIFACTS_DIR (default: artifacts)
      TOP_K (default: 5)
      N_RUNS (default: 300)
      HNSW_M (default: 16)
      HNSW_EFC (default: 200)
      HNSW_EFS (default: 80)
    """
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
    user_id = os.environ["USER_ID"]
    document_id = os.environ["DOCUMENT_ID"]
    query_text = os.environ["QUERY_TEXT"]

    k = int(os.environ.get("TOP_K", "5"))
    n_runs = int(os.environ.get("N_RUNS", "300"))

    hnsw_m = int(os.environ.get("HNSW_M", "16"))
    hnsw_efc = int(os.environ.get("HNSW_EFC", "200"))
    hnsw_efs = int(os.environ.get("HNSW_EFS", "80"))

    return artifacts_dir, user_id, document_id, query_text, k, n_runs, hnsw_m, hnsw_efc, hnsw_efs


def filter_doc_rows(
    ids: list[str],
    docstore: list[dict],
    deleted: set[str],
    document_id: str,
) -> list[int]:
    """
    Returns the corpus row indices whose metadata.document_id matches document_id
    and are not deleted.
    """
    n = min(len(ids), len(docstore))
    rows: list[int] = []
    for i in range(n):
        if ids[i] in deleted:
            continue
        md = docstore[i] or {}
        if str(md.get("document_id")) == str(document_id):
            rows.append(i)
    return rows


def main():
    artifacts_dir, user_id, document_id, query_text, k, n_runs, hnsw_m, hnsw_efc, hnsw_efs = parse_env()

    corpus_dir = artifacts_dir / f"user_{user_id}" / "_corpus"
    if not (corpus_dir / "vectors.f32").exists():
        raise FileNotFoundError(f"Missing {corpus_dir / 'vectors.f32'}")
    if not (corpus_dir / "meta.json").exists():
        raise FileNotFoundError(f"Missing {corpus_dir / 'meta.json'}")
    if not (corpus_dir / "ids.txt").exists():
        raise FileNotFoundError(f"Missing {corpus_dir / 'ids.txt'}")
    if not (corpus_dir / "docstore.jsonl").exists():
        raise FileNotFoundError(f"Missing {corpus_dir / 'docstore.jsonl'}")

    meta = load_meta(corpus_dir)
    dim = int(meta["dim"])
    ids = load_ids(corpus_dir)
    deleted = load_deleted_ids(corpus_dir)
    docstore = load_docstore(corpus_dir)
    vectors = load_vectors(corpus_dir, dim=dim)

    if len(ids) != vectors.shape[0]:
        raise ValueError(f"ids.txt count ({len(ids)}) != vectors rows ({vectors.shape[0]})")
    if len(docstore) != vectors.shape[0]:
        print(f"[WARN] docstore rows ({len(docstore)}) != vectors rows ({vectors.shape[0]}). "
              "Filtering will use min(len(ids), len(docstore)).")

    # 1) Filter user vectors down to ONLY this document
    doc_rows = filter_doc_rows(ids, docstore, deleted, document_id=document_id)
    if not doc_rows:
        raise RuntimeError(
            f"No vectors found for document_id={document_id}. "
            "Check docstore.jsonl metadata.document_id and deleted_ids.txt."
        )

    doc_ids = [ids[i] for i in doc_rows]
    doc_metas = [docstore[i] if i < len(docstore) else {} for i in doc_rows]
    doc_vectors = vectors[doc_rows, :]

    # 2) Build brute-force baseline over doc vectors only
    bf = LocalVectorStore(dim=dim, metric="cosine")
    bf.add(ids=doc_ids, vectors=doc_vectors, metas=doc_metas, compute_norms=True)

    # 3) Build HNSW index over doc vectors only (matches your "filter then ANN" approach)
    params = HNSWParams(
        M=hnsw_m,
        ef_construction=hnsw_efc,
        ef_search=hnsw_efs,
        metric="cosine",
        seed=42,
    )
    hnsw = HNSWIndex(params=params)
    hnsw.build(doc_vectors)

    # Attach vectors (your HNSW implementation expects vectors attached for distance)
    hnsw.vectors = hnsw._prepare_vectors(doc_vectors)
    hnsw.N, hnsw.dim = hnsw.vectors.shape

    # 4) Embed the provided query text
    q_emb = embed_query(query_text)
    q = np.asarray(q_emb, dtype=np.float32).reshape(dim,)

    # 5) Accuracy: recall@k (HNSW vs brute-force)
    exact = bf.search(q, k=k)
    exact_ids = [r.id for r in exact]

    approx = hnsw.search(q, k=k)
    approx_ids = [doc_ids[i] for i, _score in approx]

    rec = recall_at_k(exact_ids, approx_ids, k=k)

    # 6) Latency: benchmark this exact query repeatedly
    def run_bf():
        bf.search(q, k=k)

    def run_hnsw():
        hnsw.search(q, k=k)

    bf_bench = bench_search(run_bf, n_runs=n_runs)
    hnsw_bench = bench_search(run_hnsw, n_runs=n_runs)

    # 7) Print summary
    print("=== Brute-force vs HNSW (Doc-filtered, Query-text driven) ===")
    print(f"Artifacts: {artifacts_dir}")
    print(f"User:      {user_id}")
    print(f"Document:  {document_id}")
    print(f"Query:     {query_text}")
    print()
    print(f"Doc vectors: {len(doc_vectors)} / User vectors: {len(vectors)} | dim: {dim} | k: {k}")
    print(f"HNSW params: M={hnsw_m}, efC={hnsw_efc}, efS={hnsw_efs}")
    print(f"Recall@{k}:  {rec:.3f}")
    print()
    print("Brute-force (doc subset):")
    print(f"  QPS: {bf_bench.qps:.2f}, p50: {bf_bench.p50_ms:.3f} ms, p95: {bf_bench.p95_ms:.3f} ms")
    print("HNSW (doc subset):")
    print(f"  QPS: {hnsw_bench.qps:.2f}, p50: {hnsw_bench.p50_ms:.3f} ms, p95: {hnsw_bench.p95_ms:.3f} ms")

    speedup_p95 = (bf_bench.p95_ms / hnsw_bench.p95_ms) if hnsw_bench.p95_ms > 0 else 0.0
    print(f"\nSpeedup (p95): {speedup_p95:.2f}x")


if __name__ == "__main__":
    main()