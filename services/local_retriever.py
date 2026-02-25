from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import os
import numpy as np

from RAG_Chatbot_Backend.core.config import settings
from RAG_Chatbot_Backend.services.hnsw.hnsw_persist import load_hnsw
from RAG_Chatbot_Backend.services.hnsw.hnsw_index import HNSWIndex, HNSWParams
from RAG_Chatbot_Backend.services.hnsw.hnsw_persist import save_hnsw

def _max_neighbor_id(idx) -> int:
    mx = -1
    for node_layers in idx.neighbors:
        for layer in node_layers:
            for nb in layer:
                if nb > mx:
                    mx = nb
    return mx

def _user_artifacts_dir(user_id: str) -> Path:
    # IMPORTANT: your artifacts naming is "user_<uuid>"
    return Path(settings.ARTIFACTS_DIR) / f"user_{user_id}"


def _load_meta(corpus_dir: Path) -> dict[str, Any]:
    return json.loads((corpus_dir / "meta.json").read_text(encoding="utf-8"))


def _load_ids(corpus_dir: Path) -> list[str]:
    return (corpus_dir / "ids.txt").read_text(encoding="utf-8").splitlines()


def _load_deleted_ids(corpus_dir: Path) -> set[str]:
    p = corpus_dir / "deleted_ids.txt"
    if not p.exists():
        return set()
    txt = p.read_text(encoding="utf-8").strip()
    return set(txt.splitlines()) if txt else set()


def _load_docstore_rows(corpus_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (corpus_dir / "docstore.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_vectors(corpus_dir: Path, dim: int) -> np.ndarray:
    data = np.fromfile(corpus_dir / "vectors.f32", dtype=np.float32)
    if data.size % dim != 0:
        raise ValueError(f"vectors.f32 size {data.size} not divisible by dim={dim}")
    return data.reshape(-1, dim)


def query_user_index(user_id: str, query_embedding: list[float], top_k: int) -> dict[str, Any]:
    """
    Query the per-user HNSW corpus index and return a Pinecone-like response:
      {"matches": [{"id": <chunk_id>, "score": <float>, "metadata": <dict>}, ...]}
    """
    user_dir = _user_artifacts_dir(user_id)
    corpus_dir = user_dir / "_corpus"
    hnsw_dir = corpus_dir / "_hnsw"

    if not (corpus_dir / "vectors.f32").exists():
        # no corpus built yet
        return {"matches": []}
    if not (hnsw_dir / "hnsw_meta.json").exists():
        return {"matches": []}

    meta = _load_meta(corpus_dir)
    dim = int(meta["dim"])

    ids = _load_ids(corpus_dir)
    deleted = _load_deleted_ids(corpus_dir)
    docstore_rows = _load_docstore_rows(corpus_dir)

    vectors = _load_vectors(corpus_dir, dim=dim)

    # Load HNSW graph and attach vectors
    idx = load_hnsw(hnsw_dir)
    idx.vectors = idx._prepare_vectors(vectors)
    idx.N, idx.dim = idx.vectors.shape

    # ✅ Consistency check: graph must not reference nodes >= N
    mx = _max_neighbor_id(idx)
    if mx >= idx.N:
        # graph is stale/corrupted; rebuild
        params = getattr(idx, "params", None)
        if params is None:
            params = HNSWParams(M=16, ef_construction=200, ef_search=80, metric="cosine", seed=42)

        rebuilt = HNSWIndex(params=params)
        rebuilt.build(vectors)
        save_hnsw(rebuilt, hnsw_dir)

        # use rebuilt index
        idx = rebuilt

    # Convert query to numpy vector
    q = np.asarray(query_embedding, dtype=np.float32)
    if q.shape != (dim,):
        q = q.reshape(dim,)

    # Overfetch to compensate for deleted IDs
    overfetch = max(top_k * 3, top_k)
    results = idx.search(q, k=overfetch)

    matches = []
    for node_idx, score in results:
        chunk_id = ids[node_idx]
        if chunk_id in deleted:
            continue

        md = docstore_rows[node_idx] if node_idx < len(docstore_rows) else {}
        # Ensure metadata includes "citation" since your chat code expects it
        md = dict(md)
        md.setdefault("citation", md.get("chunk_id", chunk_id))

        matches.append({
            "id": chunk_id,
            "score": float(score),
            "metadata": md,
        })
        if len(matches) >= top_k:
            break

    return {"matches": matches}