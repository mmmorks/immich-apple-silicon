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

    def test_facial_recognition_variant_buckets_to_faces(self):
        log = "predict: 1 task(s) [facial-recognition] completed in 30ms\n"
        assert parse_ml_log(log)["tasks"]["faces"] == 1

    def test_empty_log_is_safe(self):
        result = parse_ml_log("")
        assert result["total"] == 0
        assert result["tasks"] == {"clip": 0, "faces": 0, "ocr": 0}
        assert result["latency_ms"] == {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "samples": 0}
        assert result["latency_by_task"]["clip"]["samples"] == 0

    def test_latency_has_p95_p99(self):
        lat = parse_ml_log(SAMPLE_LOG)["latency_ms"]
        # additive percentiles for tail latency
        assert "p95" in lat and "p99" in lat
        assert lat["p99"] >= lat["p95"] >= lat["p50"]

    def test_count_predict_lines(self):
        from immich_accelerator.ml_stats import count_predict_lines

        lines = [
            "predict: 1 task(s) [clip] completed in 10ms\n",
            "GET /predict\n",
            "predict: 2 task(s) [faces+ocr] completed in 20ms\n",
        ]
        assert count_predict_lines(lines) == 2


# Realistic log format (with the "<asctime> - <logger> - LEVEL - " prefix the
# service actually emits) so the parser is exercised against the real shape.
TASK_TIME_LOG = """\
2026-06-06 12:00:01,100 - src.main - INFO -   clip: 40ms
2026-06-06 12:00:01,200 - src.main - INFO -   faces: 120ms
2026-06-06 12:00:01,300 - src.main - INFO -   ocr: 47ms
2026-06-06 12:00:01,400 - src.main - INFO - predict: 3 task(s) [clip+facial-recognition+ocr] completed in 130ms
2026-06-06 12:00:02,000 - src.main - INFO -   clip: 60ms
2026-06-06 12:00:02,050 - src.main - INFO -   faces: 3 detected
2026-06-06 12:00:02,100 - src.main - INFO - predict: 1 task(s) [clip] completed in 62ms
"""


class TestLatencyByTask:
    def test_per_task_samples_and_p50(self):
        lbt = parse_ml_log(TASK_TIME_LOG)["latency_by_task"]
        # clip logged twice (40, 60) -> p50 50.0, max 60.0
        assert lbt["clip"]["samples"] == 2
        assert lbt["clip"]["p50"] == 50.0
        assert lbt["clip"]["max"] == 60.0
        # ocr once
        assert lbt["ocr"]["samples"] == 1
        assert lbt["ocr"]["p50"] == 47.0

    def test_detected_count_line_is_not_a_latency_sample(self):
        # "  faces: 3 detected" has no 'ms' suffix and must NOT be counted;
        # only the single "  faces: 120ms" line is a sample.
        lbt = parse_ml_log(TASK_TIME_LOG)["latency_by_task"]
        assert lbt["faces"]["samples"] == 1
        assert lbt["faces"]["p50"] == 120.0

    def test_aggregate_line_is_not_a_per_task_sample(self):
        # The aggregate "completed in 130ms" must not leak into per-task latency.
        lbt = parse_ml_log(TASK_TIME_LOG)["latency_by_task"]
        assert lbt["clip"]["samples"] == 2  # not 3 (the 130ms aggregate excluded)


EVENT_LOG = """\
2026-06-06 12:00:01,100 - src.models.clip - WARNING - Falling back to HF bf16 for SigLIP2
2026-06-06 12:00:01,200 - src.main - ERROR - Failed to decode image for face recognition
2026-06-06 12:00:01,300 - src.main - INFO - predict: 1 task(s) [clip] completed in 30ms
2026-06-06 12:00:01,400 - src.main - WARNING - Unloaded CLIP model (memory pressure, 400MB available, idle 35s)
2026-06-06 12:00:01,500 - src.main - CRITICAL - boom
"""


class TestParseMlEvents:
    def test_counts_errors_and_warnings(self):
        from immich_accelerator.ml_stats import parse_ml_events

        ev = parse_ml_events(EVENT_LOG)
        assert ev["errors"] == 2  # ERROR + CRITICAL
        assert ev["warnings"] == 2

    def test_recent_carries_level_and_trimmed_message_newest_last(self):
        from immich_accelerator.ml_stats import parse_ml_events

        ev = parse_ml_events(EVENT_LOG)
        assert ev["recent"][-1] == {"level": "CRITICAL", "msg": "boom"}
        assert ev["recent"][0]["level"] == "WARNING"
        assert ev["recent"][0]["msg"].startswith("Falling back to HF bf16")

    def test_recent_is_capped(self):
        from immich_accelerator.ml_stats import parse_ml_events

        ev = parse_ml_events(EVENT_LOG, recent=2)
        assert len(ev["recent"]) == 2
        assert ev["recent"][-1]["msg"] == "boom"

    def test_lowercase_error_word_does_not_count(self):
        from immich_accelerator.ml_stats import parse_ml_events

        ev = parse_ml_events("2026-06-06 12:00:01 - x - INFO - handled error gracefully\n")
        assert ev["errors"] == 0
        assert ev["warnings"] == 0

    def test_empty_log_is_safe(self):
        from immich_accelerator.ml_stats import parse_ml_events

        assert parse_ml_events("") == {"errors": 0, "warnings": 0, "recent": []}
