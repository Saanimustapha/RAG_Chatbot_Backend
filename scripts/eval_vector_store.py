import os
import json
from pathlib import Path
import numpy as np

from RAG_Chatbot_Backend.services.local_vector_store.store import LocalVectorStore
from RAG_Chatbot_Backend.services.local_vector_store.metrics import bench_search


def find_artifact_leaf(root: Path) -> Path:
    """
    Finds the directory that contains embeddings.bin and docstore.jsonl.

    Works for:
      - artifacts/user_x/doc_y/(vN/)? files
      - passing directly a doc folder
      - passing directly the leaf folder

    Strategy:
      1) If root already contains required files → return it
      2) Else search descendants for embeddings.bin and pick the 'best' match
         (prefer deepest path, then latest modified)
    """
    required = {"embeddings.bin", "docstore.jsonl"}

    # case 1: root is already a leaf
    if required.issubset({p.name for p in root.iterdir()}):
        return root

    # case 2: search
    candidates = []
    for emb in root.rglob("embeddings.bin"):
        leaf = emb.parent
        if (leaf / "docstore.jsonl").exists():
            candidates.append(leaf)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find embeddings.bin + docstore.jsonl under: {root}"
        )

    # prefer deepest (handles vN layouts), then most recently modified
    candidates.sort(key=lambda p: (len(p.parts), p.stat().st_mtime), reverse=True)
    return candidates[0]


def load_embeddings_bin(path: Path, dim: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size % dim != 0:
        raise ValueError(f"embeddings.bin size {data.size} not divisible by dim={dim}")
    return data.reshape(-1, dim)


def load_docstore(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    # You can point this to:
    # - artifacts/ (root)
    # - artifacts/user_<id>/
    # - artifacts/user_<id>/doc_<docid>/
    # - the leaf folder itself
    root = Path(os.environ.get("ARTIFACT_DIR", "artifacts"))

    dim = int(os.environ.get("EMBED_DIM", "384"))  # set correctly for your embed model
    leaf = find_artifact_leaf(root)

    emb_path = leaf / "embeddings.bin"
    docstore_path = leaf / "docstore.jsonl"

    print(f"Using artifact leaf: {leaf}")

    embs = load_embeddings_bin(emb_path, dim=dim)
    docstore = load_docstore(docstore_path)

    ids = [row["chunk_id"] for row in docstore]
    metas = docstore

    store = LocalVectorStore(dim=dim, metric="cosine")
    store.add(ids=ids, vectors=embs, metas=metas, compute_norms=True)

    # save/load test
    out_dir = leaf / "_local_store"
    store.save(out_dir)
    store2 = LocalVectorStore.load(out_dir)

    assert store2.count == store.count, "load/save count mismatch"

    # pick queries from existing vectors for sanity
    rng = np.random.default_rng(42)
    q_idx = rng.integers(0, store2.count, size=min(50, store2.count))
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

# $env:ARTIFACT_DIR="artifacts"
# $env:EMBED_DIM="384"
# python -m RAG_Chatbot_Backend.scripts.eval_vector_store