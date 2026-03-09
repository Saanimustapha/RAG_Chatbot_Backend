# services/hnsw/hnsw_store.py
"""
hnswlib-backed HNSW index.

Replacement for the old pure-Python HNSWIndex + hnsw_persist pair.
hnswlib is C++-backed (same speed tier as NumPy brute-force), which makes
brute-force vs ANN benchmarks valid and meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import hnswlib


@dataclass
class HNSWParams:
    """
    M:               max connections per layer — higher = better recall, more memory
    ef_construction: search width during build — higher = better recall, slower build
    ef_search:       search width at query time — tune for recall/speed trade-off
    metric:          "cosine" or "l2"  (hnswlib uses "cosine" or "l2" or "ip")
    seed:            random seed for reproducibility
    """
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 1000
    metric: str = "cosine"
    seed: int = 42


# Map your metric names to hnswlib space names
_SPACE = {"cosine": "cosine", "l2": "l2", "dot": "ip"}


class HNSWIndex:
    """
    Thin wrapper around hnswlib.Index that preserves the public interface
    the rest of the project uses:

        idx = HNSWIndex(params)
        idx.build(vectors)           # build from np.ndarray (N, dim)
        idx.add_vectors(vectors)     # incremental insert
        results = idx.search(q, k)  # returns List[(node_id, score)]

    Persistence:
        save_hnsw(idx, directory)
        idx = load_hnsw(directory)
    """

    def __init__(self, params: HNSWParams):
        self.params = params
        self._index: Optional[hnswlib.Index] = None
        self.N: int = 0
        self.dim: int = 0

    # ------------------------------------------------------------------
    # Build / insert
    # ------------------------------------------------------------------

    def build(self, vectors: np.ndarray) -> None:
        """Build index from scratch from a (N, dim) float32 array."""
        vectors = vectors.astype(np.float32)
        N, dim = vectors.shape
        self.dim = dim
        self.N = N

        space = _SPACE.get(self.params.metric, "cosine")
        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(
            max_elements=max(N, 1),
            ef_construction=self.params.ef_construction,
            M=self.params.M,
            random_seed=self.params.seed,
        )
        self._index.set_ef(self.params.ef_search)

        # hnswlib uses integer IDs — we use positional row indices
        self._index.add_items(vectors, ids=np.arange(N))

    def add_vectors(self, new_vectors: np.ndarray) -> list[int]:
        """
        Incrementally add vectors to an existing index.
        Returns the list of new node indices (positional row IDs).
        """
        if new_vectors is None or len(new_vectors) == 0:
            return []

        new_vectors = new_vectors.astype(np.float32)

        if self._index is None or self.N == 0:
            self.build(new_vectors)
            return list(range(self.N))

        n_new = new_vectors.shape[0]
        new_ids = np.arange(self.N, self.N + n_new)

        # Resize the index if needed (hnswlib requires explicit capacity)
        current_capacity = self._index.get_max_elements()
        if self.N + n_new > current_capacity:
            self._index.resize_index(self.N + n_new + max(n_new, 128))

        self._index.add_items(new_vectors, ids=new_ids)
        self.N += n_new
        return new_ids.tolist()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Returns List[(node_idx, score)] best-first.
        Score is cosine similarity for "cosine", negative L2 distance for "l2".
        Matches the old HNSWIndex.search() contract exactly.
        """
        if self._index is None or self.N == 0:
            return []

        q = query_vec.astype(np.float32)
        if q.ndim == 2:
            q = q[0]

        k_capped = min(k, self.N)
        labels, distances = self._index.knn_query(q, k=k_capped)
        # labels/distances are (1, k) arrays
        labels = labels[0]
        distances = distances[0]

        out: List[Tuple[int, float]] = []
        for node_idx, dist in zip(labels, distances):
            if self.params.metric == "cosine":
                # hnswlib returns 1 - cosine_similarity for "cosine" space
                score = float(1.0 - dist)
            else:
                score = float(-dist)  # negative so higher=better, matching old API
            out.append((int(node_idx), score))

        return out  # already sorted best→worst by hnswlib

    # ------------------------------------------------------------------
    # Internal helper kept for compatibility with local_retriever.py
    # (it calls idx._prepare_vectors(vectors) before assigning idx.vectors)
    # ------------------------------------------------------------------

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """No-op kept for call-site compatibility. hnswlib handles normalisation."""
        return vectors.astype(np.float32)


# ------------------------------------------------------------------
# Persistence helpers  (replaces hnsw_persist.py)
# ------------------------------------------------------------------

def save_hnsw(index: HNSWIndex, directory: str | Path) -> None:
    """
    Persist the hnswlib index to disk.
    Saves:
      hnsw.bin   — native hnswlib binary (compact, fast)
      hnsw_params.json — HNSWParams + N + dim so we can reconstruct on load
    """
    import json

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)

    if index._index is not None:
        index._index.save_index(str(d / "hnsw.bin"))

    meta = {
        "params": {
            "M": index.params.M,
            "ef_construction": index.params.ef_construction,
            "ef_search": index.params.ef_search,
            "metric": index.params.metric,
            "seed": index.params.seed,
        },
        "N": index.N,
        "dim": index.dim,
    }
    (d / "hnsw_params.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_hnsw(directory: str | Path) -> HNSWIndex:
    """
    Load a persisted hnswlib index from disk.
    Falls back gracefully if only the old pure-Python format is found
    (in that case it returns an empty index so the caller can rebuild).
    """
    import json

    d = Path(directory)
    params_path = d / "hnsw_params.json"
    bin_path = d / "hnsw.bin"

    # ── Old format present, new format absent → signal caller to rebuild ──
    old_meta = d / "hnsw_meta.json"
    if old_meta.exists() and not params_path.exists():
        # Return empty shell — callers already handle the "rebuild if stale" path
        return HNSWIndex(params=HNSWParams())

    if not params_path.exists():
        return HNSWIndex(params=HNSWParams())

    meta = json.loads(params_path.read_text(encoding="utf-8"))
    params = HNSWParams(**meta["params"])
    idx = HNSWIndex(params=params)
    idx.N = meta["N"]
    idx.dim = meta["dim"]

    if bin_path.exists() and idx.dim > 0 and idx.N > 0:
        space = _SPACE.get(params.metric, "cosine")
        inner = hnswlib.Index(space=space, dim=idx.dim)
        inner.load_index(str(bin_path), max_elements=idx.N)
        inner.set_ef(params.ef_search)
        idx._index = inner

    return idx