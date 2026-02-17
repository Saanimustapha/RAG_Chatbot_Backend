import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Metric = Literal["cosine", "dot"]

@dataclass
class StoreMeta:
    dim: int
    count: int
    metric: Metric
    has_norms: bool
    version: int = 1

def write_meta(path: Path, meta: StoreMeta) -> None:
    path.write_text(json.dumps(meta.__dict__, indent=2), encoding="utf-8")

def read_meta(path: Path) -> StoreMeta:
    d = json.loads(path.read_text(encoding="utf-8"))
    return StoreMeta(**d)
