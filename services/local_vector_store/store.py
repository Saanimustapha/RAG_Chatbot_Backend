from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from RAG_Chatbot_Backend.services.local_vector_store.format import StoreMeta, write_meta, read_meta

Metric = Literal["cosine", "dot"]

@dataclass
class SearchResult:
    id: str
    score: float
    meta: dict[str, Any] | None = None

class LocalVectorStore:
    """
    Brute-force vector store backed by a single NumPy matmul.

    For cosine metric, vectors are L2-normalised once at add() time so that
    cosine similarity reduces to a plain dot product at search time:

        score = normalised_matrix @ normalised_query   (one BLAS sgemv / sgemm)

    This matches exactly what hnswlib does internally — both methods pay the
    same per-query cost — giving a valid apples-to-apples latency comparison.
    The _norms array is kept only for reconstructing original (un-normalised)
    vectors on save/load so callers that read raw vectors back get the originals.
    """

    def __init__(self, dim: int, metric: Metric = "cosine"):
        self.dim = dim
        self.metric = metric

        # _vectors always stores the RAW (un-normalised) vectors for faithful
        # persistence; _normalised stores the unit vectors used at query time.
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._normalised: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._ids: list[str] = []
        self._metas: list[dict[str, Any] | None] = []
        self._norms: np.ndarray | None = None  # shape (n,) — kept for save/load

    @property
    def count(self) -> int:
        return self._vectors.shape[0]

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalise(mat: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalisation. Safe against zero-vectors."""
        norms = np.linalg.norm(mat, axis=1, keepdims=True).astype(np.float32)
        norms = np.where(norms < 1e-12, 1.0, norms)   # avoid div-by-zero
        return (mat / norms).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metas: list[dict[str, Any] | None] | None = None,
        compute_norms: bool = True,   # kept for call-site compatibility
    ) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"vectors must be shape (n, {self.dim})")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids length must match number of vectors")

        if metas is None:
            metas = [None] * len(ids)
        if len(metas) != len(ids):
            raise ValueError("metas length must match ids length")

        self._ids.extend(ids)
        self._metas.extend(metas)

        # Keep raw vectors for persistence
        self._vectors = np.vstack([self._vectors, vectors])

        if self.metric == "cosine":
            # Pre-normalise once here so search() is a single matmul with no
            # per-query division — identical work to hnswlib's inner loop.
            new_norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
            self._norms = (
                new_norms if self._norms is None
                else np.concatenate([self._norms, new_norms])
            )
            normed = self._l2_normalise(vectors)
            self._normalised = np.vstack([self._normalised, normed])
        else:
            # dot metric: normalised == raw
            self._normalised = self._vectors

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        if self.count == 0:
            return []

        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim == 2:
            query = query[0]
        if query.shape != (self.dim,):
            raise ValueError(f"query must be shape ({self.dim},) or (1, {self.dim})")

        if self.metric == "cosine":
            # Normalise query once, then it's a pure matmul — no division inside loop
            q = query / (float(np.linalg.norm(query)) + 1e-12)
            scores = self._normalised @ q.astype(np.float32)
        else:
            scores = self._vectors @ query

        k = min(k, self.count)
        # argpartition is O(N) — faster than full argsort for large N
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        return [
            SearchResult(id=self._ids[i], score=float(scores[i]), meta=self._metas[i])
            for i in idx
        ]

    # -------- Persistence --------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        vectors_path = d / "vectors.f32"
        ids_path = d / "ids.txt"
        metas_path = d / "metas.jsonl"
        norms_path = d / "norms.f32"
        meta_path = d / "meta.json"

        # write vectors as raw float32
        self._vectors.astype(np.float32).tofile(vectors_path)

        # ids as one per line
        ids_path.write_text("\n".join(self._ids), encoding="utf-8")

        # metas as jsonl (can be {})
        with metas_path.open("w", encoding="utf-8") as f:
            for m in self._metas:
                f.write((__import__("json").dumps(m or {})) + "\n")

        has_norms = self.metric == "cosine" and self._norms is not None
        if has_norms:
            self._norms.astype(np.float32).tofile(norms_path)

        write_meta(meta_path, StoreMeta(
            dim=self.dim,
            count=self.count,
            metric=self.metric,
            has_norms=has_norms,
        ))

    @classmethod
    def load(cls, directory: str | Path) -> "LocalVectorStore":
        import json

        d = Path(directory)
        meta = read_meta(d / "meta.json")
        store = cls(dim=meta.dim, metric=meta.metric)

        vec = np.fromfile(d / "vectors.f32", dtype=np.float32)
        if meta.count == 0:
            return store

        vec = vec.reshape(meta.count, meta.dim)
        store._vectors = vec

        # Rebuild the pre-normalised matrix from raw vectors — paid once at
        # load time so every search() call is a single matmul with no division.
        if meta.metric == "cosine":
            store._normalised = cls._l2_normalise(vec)
        else:
            store._normalised = vec

        store._ids = (d / "ids.txt").read_text(encoding="utf-8").splitlines()

        metas = []
        with (d / "metas.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                metas.append(json.loads(line) if line else {})
        store._metas = metas

        if meta.has_norms and (d / "norms.f32").exists():
            store._norms = np.fromfile(d / "norms.f32", dtype=np.float32)
        return store