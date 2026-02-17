import time
from dataclasses import dataclass

@dataclass
class BenchResult:
    qps: float
    p50_ms: float
    p95_ms: float

def bench_search(fn, n_runs: int = 200) -> BenchResult:
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times.sort()
    p50 = times[int(0.50 * (n_runs - 1))]
    p95 = times[int(0.95 * (n_runs - 1))]
    total_s = sum(times) / 1000.0
    qps = n_runs / total_s if total_s > 0 else 0.0
    return BenchResult(qps=qps, p50_ms=p50, p95_ms=p95)
