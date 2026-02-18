from __future__ import annotations

import heapq
import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class HNSWParams:
    """
    M: max neighbors per node per layer (layer 0 can use M0 = 2*M typically)
    ef_construction: search breadth during insertion (bigger -> better recall, slower build)
    ef_search: search breadth during query time (bigger -> better recall, slower query)
    metric: "cosine" (recommended) or "l2"
    """
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    metric: str = "cosine"
    seed: int = 42


class HNSWIndex:
    """
    Minimal HNSW implementation:
    - build from vectors
    - search top-k
    - supports cosine (via normalized vectors) or l2

    Stores:
    - self.vectors: float32 matrix (N, dim)
    - self.neighbors: List[Dict[layer -> List[int]]]
    """
    def __init__(self, params: HNSWParams):
        self.params = params
        random.seed(params.seed)

        self.vectors: Optional[np.ndarray] = None
        self.dim: int = 0
        self.N: int = 0

        # neighbors[node][layer] -> list of neighbor node indices
        self.neighbors: List[List[List[int]]] = []

        # level per node
        self.levels: List[int] = []

        # entry point + max layer
        self.entry_point: Optional[int] = None
        self.max_level: int = -1

        # normalization for cosine
        self._norms: Optional[np.ndarray] = None

    # -----------------------------
    # Distance / similarity
    # -----------------------------
    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        v = vectors.astype(np.float32)
        if self.params.metric == "cosine":
            # Normalize once so cosine similarity becomes dot
            norms = np.linalg.norm(v, axis=1) + 1e-12
            v = v / norms[:, None]
            self._norms = np.ones((v.shape[0],), dtype=np.float32)  # not used further
        return v

    def _dist(self, a_idx: int, b_vec: np.ndarray) -> float:
        """
        Return distance: smaller is better.
        """
        a = self.vectors[a_idx]
        if self.params.metric == "cosine":
            # cosine distance = 1 - dot (since vectors normalized)
            return float(1.0 - np.dot(a, b_vec))
        else:
            # l2 squared distance
            diff = a - b_vec
            return float(np.dot(diff, diff))

    def _dist_nodes(self, a_idx: int, b_idx: int) -> float:
        a = self.vectors[a_idx]
        b = self.vectors[b_idx]
        if self.params.metric == "cosine":
            return float(1.0 - np.dot(a, b))
        else:
            diff = a - b
            return float(np.dot(diff, diff))

    # -----------------------------
    # Level assignment
    # -----------------------------
    def _random_level(self) -> int:
        """
        Geometric distribution:
        P(level >= l) ~ exp(-l / mL)
        Common: level = floor(-ln(U) * mL)
        We use mL = 1 / ln(M).
        """
        M = max(2, self.params.M)
        mL = 1.0 / math.log(M)
        u = random.random()
        lvl = int(-math.log(u) * mL)
        return lvl

    # -----------------------------
    # Graph utilities
    # -----------------------------
    def _ensure_node_storage(self, node: int, level: int) -> None:
        # neighbors[node] is a list for each layer [0..level]
        while len(self.neighbors) <= node:
            self.neighbors.append([])
        while len(self.neighbors[node]) <= level:
            self.neighbors[node].append([])

    def _get_neighbors(self, node: int, layer: int) -> List[int]:
        if node >= len(self.neighbors) or layer >= len(self.neighbors[node]):
            return []
        return self.neighbors[node][layer]

    def _set_neighbors(self, node: int, layer: int, nbrs: List[int]) -> None:
        self._ensure_node_storage(node, layer)
        self.neighbors[node][layer] = nbrs

    # -----------------------------
    # Heuristic neighbor selection
    # -----------------------------
    def _select_neighbors_heuristic(
        self,
        candidates: List[int],
        new_node: int,
        M: int,
        layer: int,
    ) -> List[int]:
        """
        HNSW heuristic:
        Keep neighbors that are close to new_node but also diverse.
        A simple version:
          - sort candidates by distance(new_node, cand)
          - greedily accept cand if it is not closer to any already chosen neighbor than to new_node
        """
        # sort by distance to new_node
        candidates = sorted(candidates, key=lambda c: self._dist_nodes(new_node, c))
        selected: List[int] = []

        for c in candidates:
            if len(selected) >= M:
                break
            ok = True
            d_nc = self._dist_nodes(new_node, c)
            for s in selected:
                # if candidate is "shadowed" by an existing selected neighbor, skip
                if self._dist_nodes(s, c) < d_nc:
                    ok = False
                    break
            if ok:
                selected.append(c)
        return selected

    # -----------------------------
    # Search within a layer (used for insertion + query)
    # -----------------------------
    def _search_layer(self, q_vec: np.ndarray, entry: int, ef: int, layer: int) -> List[int]:
        """
        Best-first search in a given layer.
        Returns up to ef closest nodes found.
        """
        visited = set([entry])

        # min-heap for candidates by distance
        candidates: List[Tuple[float, int]] = [(self._dist(entry, q_vec), entry)]
        # max-heap for results (store negative distance to simulate max heap)
        results: List[Tuple[float, int]] = [(-self._dist(entry, q_vec), entry)]

        while candidates:
            dist_c, c = heapq.heappop(candidates)

            worst_dist = -results[0][0]  # max-heap top is worst (largest dist)
            if dist_c > worst_dist:
                # No better candidates possible
                break

            for nb in self._get_neighbors(c, layer):
                if nb in visited:
                    continue
                visited.add(nb)
                d = self._dist(nb, q_vec)

                if len(results) < ef or d < worst_dist:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(results, (-d, nb))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst_dist = -results[0][0]

        # return results sorted best->worst
        out = [n for _, n in sorted([(-d, n) for d, n in results], key=lambda x: x[0])]
        return out

    # -----------------------------
    # Build / insert
    # -----------------------------
    def build(self, vectors: np.ndarray) -> None:
        """
        Build index from scratch.
        """
        self.vectors = self._prepare_vectors(vectors)
        self.N, self.dim = self.vectors.shape

        self.neighbors = []
        self.levels = []
        self.entry_point = None
        self.max_level = -1

        for i in range(self.N):
            self.insert(i)

    def insert(self, node: int) -> None:
        """
        Insert node into HNSW graph.
        Assumes self.vectors already set.
        """
        lvl = self._random_level()
        self.levels.append(lvl)
        self._ensure_node_storage(node, lvl)

        if self.entry_point is None:
            # first node
            self.entry_point = node
            self.max_level = lvl
            return

        ep = self.entry_point
        cur = ep

        # 1) greedy navigate from top layer down to target level
        for layer in range(self.max_level, lvl, -1):
            # greedy: move to neighbor that improves distance
            changed = True
            while changed:
                changed = False
                cur_dist = self._dist(cur, self.vectors[node])
                for nb in self._get_neighbors(cur, layer):
                    d = self._dist(nb, self.vectors[node])
                    if d < cur_dist:
                        cur_dist = d
                        cur = nb
                        changed = True

        # 2) for layers min(lvl, max_level) down to 0: search + connect
        for layer in range(min(lvl, self.max_level), -1, -1):
            efc = self.params.ef_construction
            candidates = self._search_layer(self.vectors[node], cur, ef=efc, layer=layer)

            # connect using heuristic and cap degree
            M = self.params.M
            M_layer = (2 * M) if layer == 0 else M

            selected = self._select_neighbors_heuristic(candidates, node, M_layer, layer)
            self._set_neighbors(node, layer, selected)

            # bidirectional link + prune neighbors of existing nodes
            for nb in selected:
                nbs = self._get_neighbors(nb, layer)
                if node not in nbs:
                    nbs = nbs + [node]

                # prune if too many
                if len(nbs) > M_layer:
                    nbs = self._select_neighbors_heuristic(nbs, nb, M_layer, layer)
                self._set_neighbors(nb, layer, nbs)

            # choose new entry for next lower layer as closest among selected/candidates
            if candidates:
                cur = min(candidates, key=lambda c: self._dist_nodes(c, node))

        # 3) update entry point if node has new max level
        if lvl > self.max_level:
            self.max_level = lvl
            self.entry_point = node

    # -----------------------------
    # Query
    # -----------------------------
    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Returns list of (node_idx, score) best->worst.
        Score returned is similarity for cosine, negative distance for l2.
        """
        if self.entry_point is None or self.vectors is None or self.N == 0:
            return []

        q = query_vec.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        q = q[0]

        if self.params.metric == "cosine":
            qn = np.linalg.norm(q) + 1e-12
            q = q / qn

        cur = self.entry_point

        # greedy descent from top layer
        for layer in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                cur_dist = self._dist(cur, q)
                for nb in self._get_neighbors(cur, layer):
                    d = self._dist(nb, q)
                    if d < cur_dist:
                        cur_dist = d
                        cur = nb
                        changed = True

        # efSearch exploration at layer 0
        ef = max(self.params.ef_search, k)
        found = self._search_layer(q, cur, ef=ef, layer=0)

        # rank by exact distance
        found = sorted(found, key=lambda n: self._dist(n, q))[:k]

        # convert to score (higher is better)
        out: List[Tuple[int, float]] = []
        for n in found:
            d = self._dist(n, q)
            if self.params.metric == "cosine":
                sim = 1.0 - d
                out.append((n, float(sim)))
            else:
                out.append((n, float(-d)))
        return out
