from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from RAG_Chatbot_Backend.core.config import settings
from RAG_Chatbot_Backend.services.hnsw.hnsw_store import load_hnsw, save_hnsw, HNSWIndex, HNSWParams




def _user_artifacts_dir(user_id: str) -> Path:
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


def _normalize_rows_to_str_set(values: Optional[Iterable[str]]) -> Optional[set[str]]:
    if not values:
        return None
    return {str(v) for v in values}


def _cosine_topk(
    q: np.ndarray,
    mat: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (top_indices, top_scores) where:
      - top_indices are indices into mat
      - top_scores are cosine similarity scores
    Assumes vectors are already normalized (recommended).
    If not normalized, we normalize q and mat row-wise for cosine.
    """
    if mat.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # Defensive normalize (cheap enough for subsets)
    qn = q / (np.linalg.norm(q) + 1e-12)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

    scores = mn @ qn  # (N,)
    k = min(k, scores.shape[0])

    # partial selection then sort
    idx_part = np.argpartition(-scores, kth=k - 1)[:k]
    idx_sorted = idx_part[np.argsort(-scores[idx_part])]
    return idx_sorted.astype(np.int64), scores[idx_sorted].astype(np.float32)


def query_user_index(
    user_id: str,
    query_embedding: list[float],
    top_k: int,
    document_ids: Optional[list[str]] = None,
    source_type: Optional[str] = None,
    filename_contains: Optional[str] = None,
) -> dict[str, Any]:
    """
    Query per-user corpus.

    If document_ids is provided:
      - filter corpus rows to those docs first
      - run cosine similarity ONLY on that subset
      - return top_k

    If document_ids is not provided:
      - use HNSW over the entire corpus (existing behavior)

    Returns Pinecone-like response:
      {"matches": [{"id": <chunk_id>, "score": <float>, "metadata": <dict>}, ...]}
    """
    user_dir = _user_artifacts_dir(user_id)
    corpus_dir = user_dir / "_corpus"
    hnsw_dir = corpus_dir / "_hnsw"

    if not (corpus_dir / "vectors.f32").exists():
        return {"matches": []}
    if not (corpus_dir / "meta.json").exists():
        return {"matches": []}

    meta = _load_meta(corpus_dir)
    dim = int(meta["dim"])

    ids = _load_ids(corpus_dir)
    deleted = _load_deleted_ids(corpus_dir)
    docstore_rows = _load_docstore_rows(corpus_dir)
    vectors = _load_vectors(corpus_dir, dim=dim)

    q = np.asarray(query_embedding, dtype=np.float32)
    if q.shape != (dim,):
        q = q.reshape(dim,)

    allowed_docs = _normalize_rows_to_str_set(document_ids)
    st_filter = (source_type or "").strip().lower() or None
    fn_filter = (filename_contains or "").strip().lower() or None

    # ------------------------------------------------------------------
    # STRICT DOC-FILTERED SEARCH (subset cosine)
    # ------------------------------------------------------------------
    if allowed_docs:
        candidate_rows: list[int] = []
        candidate_vecs: list[np.ndarray] = []

        for i, chunk_id in enumerate(ids):
            if chunk_id in deleted:
                continue
            if i >= len(docstore_rows):
                continue

            md = docstore_rows[i] or {}
            md_doc = str(md.get("document_id") or "")
            if md_doc not in allowed_docs:
                continue

            if st_filter and (str(md.get("source_type") or "").lower() != st_filter):
                continue
            if fn_filter:
                fn = str(md.get("filename") or "").lower()
                if fn_filter not in fn:
                    continue

            candidate_rows.append(i)
            candidate_vecs.append(vectors[i])

        if not candidate_rows:
            return {"matches": []}

        mat = np.stack(candidate_vecs, axis=0)  # (Ncand, dim)
        top_local_idx, top_scores = _cosine_topk(q, mat, top_k)

        matches: list[dict[str, Any]] = []
        for local_pos, score in zip(top_local_idx.tolist(), top_scores.tolist()):
            global_row = candidate_rows[local_pos]
            chunk_id = ids[global_row]
            md = docstore_rows[global_row] if global_row < len(docstore_rows) else {}
            md = dict(md or {})
            md.setdefault("citation", md.get("chunk_id", chunk_id))
            matches.append({
                "id": chunk_id,
                "score": float(score),
                "metadata": md,
            })

        return {"matches": matches}

    # ------------------------------------------------------------------
    # GLOBAL SEARCH (HNSW) — unchanged behavior
    # ------------------------------------------------------------------
    if not (hnsw_dir / "hnsw_meta.json").exists():
        return {"matches": []}

    idx = load_hnsw(hnsw_dir)

    # Rebuild if index is empty 
    if idx._index is None or idx.N == 0:
        params = idx.params or HNSWParams(M=16, ef_construction=200, ef_search=80, metric="cosine", seed=42)
        idx = HNSWIndex(params=params)
        idx.build(vectors)
        save_hnsw(idx, hnsw_dir)

    overfetch = max(top_k * 3, top_k)
    results = idx.search(q, k=overfetch)

    matches = []
    for node_idx, score in results:
        chunk_id = ids[node_idx]
        if chunk_id in deleted:
            continue

        md = docstore_rows[node_idx] if node_idx < len(docstore_rows) else {}
        md = dict(md or {})
        md.setdefault("citation", md.get("chunk_id", chunk_id))

        matches.append({
            "id": chunk_id,
            "score": float(score),
            "metadata": md,
        })
        if len(matches) >= top_k:
            break

    return {"matches": matches}