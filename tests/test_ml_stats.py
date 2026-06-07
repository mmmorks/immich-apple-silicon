"""Tests for immich_accelerator.ml_stats — ML request-log parsing."""
from __future__ import annotations

from immich_accelerator.ml_stats import parse_ml_log

SAMPLE_LOG = """\
2026-06-06 12:00:01 INFO predict: 3 task(s) [clip+faces+ocr] completed in 142ms
2026-06-06 12:00:02 INFO predict: 1 task(s) [clip] completed in 38ms
GET /predict
2026-06-06 12:00:03 INFO predict: 2 task(s) [faces+ocr] completed in 88ms
some unrelated log line
predict: 1 task(s) [clip] completed in 50ms
"""


class TestParseMlLog:
    def test_counts_total_predicts(self):
        assert parse_ml_log(SAMPLE_LOG)["total"] == 4

    def test_buckets_tasks_by_substring(self):
        tasks = parse_ml_log(SAMPLE_LOG)["tasks"]
        assert tasks == {"clip": 3, "faces": 2, "ocr": 2}

    def test_latency_p50_and_samples(self):
        lat = parse_ml_log(SAMPLE_LOG)["latency_ms"]
        # ms values [142, 38, 88, 50] -> sorted [38, 50, 88, 142], median 69.0
        assert lat["p50"] == 69.0
        assert lat["max"] == 142.0
        assert lat["samples"] == 4

    def test_empty_log_is_safe(self):
        result = parse_ml_log("")
        assert result["total"] == 0
        assert result["tasks"] == {"clip": 0, "faces": 0, "ocr": 0}
        assert result["latency_ms"] == {"p50": 0.0, "max": 0.0, "samples": 0}
