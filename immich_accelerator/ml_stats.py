"""Pure parser for the immich-ml-metal request log.

The ML service logs one line per inference at INFO:

    predict: 3 task(s) [clip+faces+ocr] completed in 142ms

We parse those lines into counts (total + per-task) and latency stats.
Task bucketing is by substring so it survives naming variants
(``faces`` vs ``facial-recognition`` etc.). Pure function — no I/O —
so it is trivially unit-testable against a captured fixture.

NOTE: the leading timestamp/level prefix is produced by the service's
logger and is ignored here; only the ``predict: ... completed in Nms``
portion matters. Verify the suffix shape against a real log (Task 13).
"""
from __future__ import annotations

import re
import statistics

_PREDICT_RE = re.compile(
    r"predict:\s+\d+\s+task\(s\)\s+\[([^\]]+)\]\s+completed in\s+([\d.]+)\s*ms"
)


def parse_ml_log(text: str) -> dict:
    tasks = {"clip": 0, "faces": 0, "ocr": 0}
    latencies: list[float] = []
    total = 0
    for names, ms in _PREDICT_RE.findall(text):
        total += 1
        latencies.append(float(ms))
        lowered = names.lower()
        if "clip" in lowered:
            tasks["clip"] += 1
        if "fac" in lowered:
            tasks["faces"] += 1
        if "ocr" in lowered:
            tasks["ocr"] += 1
    if latencies:
        latency_ms = {
            "p50": round(statistics.median(latencies), 1),
            "max": round(max(latencies), 1),
            "samples": len(latencies),
        }
    else:
        latency_ms = {"p50": 0.0, "max": 0.0, "samples": 0}
    return {"total": total, "tasks": tasks, "latency_ms": latency_ms}
