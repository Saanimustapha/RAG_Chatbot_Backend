from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hnsw_index import HNSWIndex, HNSWParams


def save_hnsw(index: HNSWIndex, directory: str | Path) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)

    # meta (params + entry/max)
    meta = {
        "params": index.params.__dict__,
        "entry_point": index.entry_point,
        "max_level": index.max_level,
        "levels": index.levels,
        "N": index.N,
        "dim": index.dim,
    }
    (d / "hnsw_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # neighbors graph as jsonl: one line per node
    # {"node": i, "layers": [[...layer0...],[...layer1...], ...]}
    with (d / "hnsw_graph.jsonl").open("w", encoding="utf-8") as f:
        for i, layers in enumerate(index.neighbors):
            f.write(json.dumps({"node": i, "layers": layers}) + "\n")


def load_hnsw(directory: str | Path) -> HNSWIndex:
    d = Path(directory)
    meta = json.loads((d / "hnsw_meta.json").read_text(encoding="utf-8"))

    params = HNSWParams(**meta["params"])
    idx = HNSWIndex(params=params)

    idx.entry_point = meta["entry_point"]
    idx.max_level = meta["max_level"]
    idx.levels = meta["levels"]
    idx.N = meta["N"]
    idx.dim = meta["dim"]

    # load graph
    neighbors = []
    with (d / "hnsw_graph.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            neighbors.append(row["layers"])
    idx.neighbors = neighbors

    return idx
