import json
import numpy as np
from pathlib import Path
from typing import Any, Optional

from RAG_Chatbot_Backend.core.config import settings
from RAG_Chatbot_Backend.services.hnsw.hnsw_store import load_hnsw
from RAG_Chatbot_Backend.services.bm25.bm25_store import UserBM25
from RAG_Chatbot_Backend.services.retrieval.filters import allowed_rows_for_docs

def _user_corpus_dir(user_id: str) -> Path:
    return Path(settings.ARTIFACTS_DIR) / f"user_{user_id}" / "_corpus"

def _load_meta(corpus_dir: Path) -> dict:
    return json.loads((corpus_dir / "meta.json").read_text(encoding="utf-8"))

def _load_ids(corpus_dir: Path) -> list[str]:
    return (corpus_dir / "ids.txt").read_text(encoding="utf-8").splitlines()

def _load_docstore_rows(corpus_dir: Path) -> list[dict[str, Any]]:
    rows = []
    with (corpus_dir / "docstore.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _load_vectors(corpus_dir: Path, dim: int) -> np.ndarray:
    data = np.fromfile(corpus_dir / "vectors.f32", dtype=np.float32)
    return data.reshape(-1, dim)

def _minmax_norm(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

def hybrid_retrieve(
    *,
    user_id: str,
    query_emb: list[float],
    query_text: str,
    top_k: int,
    document_ids: Optional[list[str]] = None,
    w_sem: float = 0.65,
    w_kw: float = 0.35,
    overfetch: int = 200,
) -> list[dict[str, Any]]:
    corpus_dir = _user_corpus_dir(user_id)
    hnsw_dir = corpus_dir / "_hnsw"
    meta = _load_meta(corpus_dir)
    dim = int(meta["dim"])

    ids = _load_ids(corpus_dir)
    docstore = _load_docstore_rows(corpus_dir)

    allowed_rows = None
    if document_ids:
        allowed_rows = allowed_rows_for_docs(corpus_dir, document_ids)

    # ---- semantic candidates (HNSW global, then filter rows)
    vectors = _load_vectors(corpus_dir, dim)
    idx = load_hnsw(hnsw_dir)
    idx.vectors = idx._prepare_vectors(vectors)
    idx.N, idx.dim = idx.vectors.shape

    q = np.asarray(query_emb, dtype=np.float32).reshape(dim,)
    sem = idx.search(q, k=min(overfetch, idx.N))
    sem_scores = {}
    for row, score in sem:
        if allowed_rows is not None and row not in allowed_rows:
            continue
        sem_scores[row] = float(score)

    # ---- keyword candidates (BM25), already supports allowed_rows
    bm25 = UserBM25(corpus_dir)
    kw = bm25.search(query_text, top_k=overfetch, allowed_rows=allowed_rows)
    kw_scores = {row: score for row, score in kw}

    # normalize and merge
    sem_n = _minmax_norm(sem_scores)
    kw_n = _minmax_norm(kw_scores)

    all_rows = set(sem_n.keys()) | set(kw_n.keys())
    merged = []
    for r in all_rows:
        score = w_sem * sem_n.get(r, 0.0) + w_kw * kw_n.get(r, 0.0)
        md = docstore[r] if r < len(docstore) else {}
        merged.append({
            "corpus_row": r,
            "id": ids[r] if r < len(ids) else str(r),
            "score": float(score),
            "source": "hybrid",
            "metadata": md,
        })

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k * 5]  # return candidates for rerank/diversity