import numpy as np
from typing import Any

def mmr_select(
    candidates: list[dict[str, Any]],
    vectors: np.ndarray,              # full corpus vectors
    top_k: int,
    lambda_mult: float = 0.7,
    sim_threshold: float = 0.92,
) -> list[dict[str, Any]]:
    """
    MMR-style selection to avoid near-duplicates.
    Requires candidates include "corpus_row".
    """
    if not candidates:
        return []

    selected = []
    selected_rows = []

    cand_rows = [c["corpus_row"] for c in candidates]
    cand_vecs = vectors[cand_rows]

    # normalize
    cand_vecs = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12)

    # start with best
    selected.append(candidates[0])
    selected_rows.append(0)

    while len(selected) < top_k and len(selected_rows) < len(candidates):
        best_idx = None
        best_score = -1e9

        for i in range(len(candidates)):
            if i in selected_rows:
                continue

            relevance = candidates[i]["score"]

            # max similarity to already selected
            sims = cand_vecs[i] @ cand_vecs[selected_rows].T
            max_sim = float(np.max(sims)) if len(selected_rows) else 0.0

            # hard dedup cutoff
            if max_sim >= sim_threshold:
                continue

            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx is None:
            break
        selected.append(candidates[best_idx])
        selected_rows.append(best_idx)

    return selected