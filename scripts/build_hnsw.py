import os
from pathlib import Path
import numpy as np

from RAG_Chatbot_Backend.services.local_vector_store.store import LocalVectorStore
from RAG_Chatbot_Backend.services.hnsw.hnsw_index import HNSWIndex, HNSWParams
from RAG_Chatbot_Backend.services.hnsw.hnsw_persist import save_hnsw


def main():
    leaf = Path(os.environ["ARTIFACT_LEAF"])  # point to doc folder with embeddings.bin/docstore.jsonl
    dim = int(os.environ.get("EMBED_DIM", "384"))

    # Load baseline store (Phase 3 format)
    # We'll reuse its vector loading logic by pointing it at your saved store, OR read embeddings.bin directly.
    emb_path = leaf / "embeddings.bin"
    vectors = np.fromfile(emb_path, dtype=np.float32)
    vectors = vectors.reshape(-1, dim)

    params = HNSWParams(
        M=int(os.environ.get("HNSW_M", "16")),
        ef_construction=int(os.environ.get("HNSW_EFC", "200")),
        ef_search=int(os.environ.get("HNSW_EFS", "50")),
        metric=os.environ.get("HNSW_METRIC", "cosine"),
        seed=42,
    )

    index = HNSWIndex(params=params)
    index.build(vectors)

    out_dir = leaf / "_hnsw_index"
    save_hnsw(index, out_dir)
    print(f"Saved HNSW index to: {out_dir}")


if __name__ == "__main__":
    main()
