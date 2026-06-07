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

_PREDICT_RE = re.compile(r"predict:\s+\d+\s+task\(s\)\s+\[([^\]]+)\]\s+completed in\s+([\d.]+)\s*ms")

# Per-task timing lines the service logs alongside the aggregate, e.g.
# "  clip: 47ms" (src.main._timed). The "  faces: N detected" count line has no
# "ms" suffix so it is not matched; the lookbehind avoids matching a longer word
# that happens to end in a task name (e.g. "subclip:").
_TASK_TIME_RE = re.compile(r"(?<![A-Za-z])(clip|faces|ocr):\s+([\d.]+)\s*ms\b")

# Log-level token for the error/warning feed. Case-sensitive, so a lowercase
# "error" inside a normal message does not count.
_LEVEL_RE = re.compile(r"\b(ERROR|CRITICAL|WARNING)\b")


def _pct(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in 0..100), numpy-free.

    Matches numpy's default 'linear' method, so p50 equals the median.
    """
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return round(s[0], 1)
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (pos - lo), 1)


def _latency_stats(values: list[float]) -> dict:
    """p50/p95/p99/max/samples for a list of latency samples (ms)."""
    return {
        "p50": _pct(values, 50),
        "p95": _pct(values, 95),
        "p99": _pct(values, 99),
        "max": round(max(values), 1) if values else 0.0,
        "samples": len(values),
    }


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
    # Per-task latency from the per-task timing lines ("  clip: 47ms"), distinct
    # from the aggregate "completed in Nms" counted above.
    per_task: dict[str, list[float]] = {"clip": [], "faces": [], "ocr": []}
    for name, ms in _TASK_TIME_RE.findall(text):
        per_task[name].append(float(ms))

    return {
        "total": total,
        "tasks": tasks,
        "latency_ms": _latency_stats(latencies),
        "latency_by_task": {k: _latency_stats(v) for k, v in per_task.items()},
    }


def count_predict_lines(lines) -> int:
    """Count predict-completion lines in an iterable of log lines.

    Used to derive a cumulative, monotonic predict count by streaming the
    whole log file — the windowed ``parse_ml_log`` total is NOT cumulative.
    """
    return sum(1 for line in lines if _PREDICT_RE.search(line))


def parse_ml_events(text: str, recent: int = 8) -> dict:
    """Count ERROR/CRITICAL/WARNING log lines and capture the most recent ones.

    Pure function over the log tail. Surfaces failures and lifecycle events
    (model fallbacks, memory-pressure unloads) that a throughput number alone
    hides. Returns ``{"errors", "warnings", "recent": [{"level", "msg"}, ...]}``
    with the newest events last. ``recent`` caps how many are retained.

    Level detection is a substring heuristic (the log prefix format varies); a
    lowercase "error" in a message does not count because the pattern is
    case-sensitive.
    """
    errors = 0
    warnings = 0
    events: list[dict] = []
    for line in text.splitlines():
        m = _LEVEL_RE.search(line)
        if not m:
            continue
        level = m.group(1)
        if level == "WARNING":
            warnings += 1
        else:
            errors += 1
        # Drop the "<asctime> - <logger> - LEVEL - " prefix for display.
        msg = line.split(" - ", 3)[-1].strip() or line.strip()
        events.append({"level": level, "msg": msg})
    return {"errors": errors, "warnings": warnings, "recent": events[-recent:]}
