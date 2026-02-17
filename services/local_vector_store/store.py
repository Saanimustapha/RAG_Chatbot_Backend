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
    Simple brute-force vector store:
    - holds vectors in memory (float32)
    - optional norms for cosine
    - persistence to disk in a custom format
    """
    def __init__(self, dim: int, metric: Metric = "cosine"):
        self.dim = dim
        self.metric = metric

        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._ids: list[str] = []
        self._metas: list[dict[str, Any] | None] = []

        self._norms: np.ndarray | None = None  # shape: (n,)

    @property
    def count(self) -> int:
        return self._vectors.shape[0]

    def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metas: list[dict[str, Any] | None] | None = None,
        compute_norms: bool = True,
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

        # append
        self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(ids)
        self._metas.extend(metas)

        if self.metric == "cosine" and compute_norms:
            new_norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
            self._norms = new_norms if self._norms is None else np.concatenate([self._norms, new_norms])

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        if self.count == 0:
            return []

        if query.dtype != np.float32:
            query = query.astype(np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape != (1, self.dim):
            raise ValueError(f"query must be shape (1, {self.dim}) or ({self.dim},)")

        q = query[0]

        if self.metric == "dot":
            scores = self._vectors @ q
        else:
            # cosine = dot / (||a|| * ||b||)
            if self._norms is None:
                self._norms = np.linalg.norm(self._vectors, axis=1).astype(np.float32)
            qn = float(np.linalg.norm(q)) + 1e-12
            scores = (self._vectors @ q) / ((self._norms * qn) + 1e-12)

        # top-k
        k = min(k, self.count)
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
        d = Path(directory)

        meta = read_meta(d / "meta.json")

        store = cls(dim=meta.dim, metric=meta.metric)

        # vectors
        vec = np.fromfile(d / "vectors.f32", dtype=np.float32)
        if meta.count == 0:
            return store

        vec = vec.reshape(meta.count, meta.dim)
        store._vectors = vec

        # ids
        store._ids = (d / "ids.txt").read_text(encoding="utf-8").splitlines()

        # metas
        metas = []
        import json
        with (d / "metas.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    metas.append({})
                else:
                    metas.append(json.loads(line))
        store._metas = metas

        # norms
        if meta.has_norms and (d / "norms.f32").exists():
            store._norms = np.fromfile(d / "norms.f32", dtype=np.float32)
        return store
