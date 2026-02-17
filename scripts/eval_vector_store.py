import os
import json
from pathlib import Path
import numpy as np

from RAG_Chatbot_Backend.services.local_vector_store.store import LocalVectorStore
from RAG_Chatbot_Backend.services.local_vector_store.metrics import bench_search

def load_embeddings_bin(path: Path, dim: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(-1, dim)

def load_docstore(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    # point this to one of your artifact directories:
    # artifacts/user_<id>/doc_<docid>/v<version>/
    artifact_dir = Path(os.environ.get("ARTIFACT_DIR", "artifacts"))
    dim = int(os.environ.get("EMBED_DIM", "384"))  # set to your embed model dim

    emb_path = artifact_dir / "embeddings.bin"
    docstore_path = artifact_dir / "docstore.jsonl"

    if not emb_path.exists() or not docstore_path.exists():
        raise SystemExit("ARTIFACT_DIR must contain embeddings.bin and docstore.jsonl")

    embs = load_embeddings_bin(emb_path, dim=dim)
    docstore = load_docstore(docstore_path)

    ids = [row["chunk_id"] for row in docstore]
    metas = docstore

    store = LocalVectorStore(dim=dim, metric="cosine")
    store.add(ids=ids, vectors=embs, metas=metas, compute_norms=True)

    # save/load test
    out_dir = artifact_dir / "_local_store"
    store.save(out_dir)
    store2 = LocalVectorStore.load(out_dir)

    assert store2.count == store.count, "load/save count mismatch"

    # pick queries from existing vectors (simple sanity)
    rng = np.random.default_rng(42)
    q_idx = rng.integers(0, store2.count, size=50)
    queries = store2._vectors[q_idx]

    # consistency check
    first = store2.search(queries[0], k=5)
    second = store2.search(queries[0], k=5)
    assert [r.id for r in first] == [r.id for r in second], "non-deterministic results"

    # benchmark
    qi = 0
    def run_one():
        nonlocal qi
        q = queries[qi % len(queries)]
        store2.search(q, k=5)
        qi += 1

    bench = bench_search(run_one, n_runs=300)
    print("LocalVectorStore benchmark")
    print(f"Vectors: {store2.count}, dim: {dim}")
    print(f"QPS: {bench.qps:.2f}")
    print(f"p50 ms: {bench.p50_ms:.3f}")
    print(f"p95 ms: {bench.p95_ms:.3f}")

if __name__ == "__main__":
    main()
