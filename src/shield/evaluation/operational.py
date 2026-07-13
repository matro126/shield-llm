from __future__ import annotations

import math


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def summarize_latency(latencies: list[float]) -> dict[str, float]:
    total = sum(latencies)
    return {
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
        "latency_mean_s": (total / len(latencies)) if latencies else 0.0,
        "throughput_req_s": (len(latencies) / total) if total > 0 else 0.0,
    }


def operational_metrics(
    latencies: list[float], vram_peak_bytes: float | None = None
) -> dict[str, float]:
    out = summarize_latency(latencies)
    if vram_peak_bytes is not None:
        out["vram_peak_gb"] = vram_peak_bytes / 1e9
    return out
