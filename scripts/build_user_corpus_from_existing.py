import os
import shutil
from pathlib import Path
import numpy as np

from RAG_Chatbot_Backend.services.corpus.corpus_store import UserCorpusStore
from RAG_Chatbot_Backend.services.hnsw.hnsw_index import HNSWIndex, HNSWParams
from RAG_Chatbot_Backend.services.hnsw.hnsw_persist import save_hnsw


def load_corpus_vectors(corpus_dir: Path, dim: int) -> np.ndarray:
    vec_path = corpus_dir / "vectors.f32"
    data = np.fromfile(vec_path, dtype=np.float32)
    if data.size % dim != 0:
        raise ValueError(f"vectors.f32 size {data.size} not divisible by dim={dim}")
    return data.reshape(-1, dim)


def main():
    user_dir = Path(os.environ["USER_ARTIFACT_DIR"])  # artifacts/user_<id>
    dim = int(os.environ.get("EMBED_DIM", "384"))

    params = HNSWParams(
        M=int(os.environ.get("HNSW_M", "16")),
        ef_construction=int(os.environ.get("HNSW_EFC", "200")),
        ef_search=int(os.environ.get("HNSW_EFS", "80")),
        metric=os.environ.get("HNSW_METRIC", "cosine"),
        seed=42,
    )

    # 1) Wipe existing corpus completely to avoid graph/vector mismatch
    corpus_dir = user_dir / "_corpus"
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)

    # 2) Recreate corpus store
    store = UserCorpusStore(user_dir=user_dir, dim=dim, metric=params.metric)

    # 3) For each doc_<id>, pick latest v<version> leaf and append
    manifest = {"docs": {}}
    for doc_dir in user_dir.glob("doc_*"):
        versions = sorted(
            doc_dir.glob("v*"),
            key=lambda p: int(p.name.replace("v", "")) if p.name.replace("v", "").isdigit() else -1
        )
        if not versions:
            continue
        latest = versions[-1]

        # Append to corpus
        start, n, chunk_ids = store.append_doc_leaf(latest)

        # Signature (from chunks.jsonl first row)
        checksum = None
        doc_version = None
        chunks_path = latest / "chunks.jsonl"
        if chunks_path.exists():
            import json
            with chunks_path.open("r", encoding="utf-8") as f:
                first = json.loads(next(f))
                checksum = first.get("checksum")
                doc_version = first.get("doc_version")

        doc_id = doc_dir.name.replace("doc_", "")
        manifest["docs"][doc_id] = {
            "checksum": checksum,
            "doc_version": doc_version,
            "active_chunk_ids": chunk_ids,
            "corpus_start": start,
            "corpus_count": n,
        }

    store.save_manifest(manifest)

    # 4) Build HNSW once from merged vectors
    vectors = load_corpus_vectors(store.corpus_dir, dim=dim)
    idx = HNSWIndex(params=params)
    idx.build(vectors)

    # 5) Save HNSW
    out_dir = store.corpus_dir / "_hnsw"
    save_hnsw(idx, out_dir)

    print(f"✅ Built per-user corpus: {store.corpus_dir}")
    print(f"✅ Built per-user HNSW:   {out_dir}")
    print(f"Vectors: {vectors.shape[0]} | dim: {vectors.shape[1]}")


if __name__ == "__main__":
    main()


# $env:USER_ARTIFACT_DIR="artifacts\user_<id>"
# $env:EMBED_DIM="384"
# python RAG_Chatbot_Backend/scripts/build_user_corpus_from_existing.py