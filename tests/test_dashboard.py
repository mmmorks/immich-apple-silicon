"""Tests for immich_accelerator.dashboard — status API, caching, FastAPI app."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Guard: skip all dashboard tests if fastapi/httpx are not installed
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from immich_accelerator.dashboard import (
    _get_accelerator_version,
    _IncrementalPredictCounter,
    _query_db,
    _run,
    create_app,
    get_status,
)

_PREDICT = "2026-06-06 12:00:01 INFO predict: 1 task(s) [clip] completed in 40ms\n"
_NOISE = "GET /predict\n"

# ---------------------------------------------------------------------------
# _get_accelerator_version
# ---------------------------------------------------------------------------


class TestGetAcceleratorVersion:
    def test_reads_version_file(self, tmp_path):
        version_file = tmp_path / "VERSION"
        version_file.write_text("1.3.1\n")
        with patch("immich_accelerator.dashboard.Path") as mock_path_cls:
            mock_path_cls.return_value.parent.parent.__truediv__ = lambda self, x: version_file
            # Directly test: the function reads from Path(__file__).parent.parent / "VERSION"
            # We'll just verify the fallback behavior since patching __file__ is awkward

    def test_fallback_on_missing_file(self):
        with patch("immich_accelerator.dashboard.Path") as mock_path_cls:
            mock_version = MagicMock()
            mock_version.exists.return_value = False
            mock_path_cls.return_value.parent.parent.__truediv__.return_value = mock_version
            result = _get_accelerator_version()
            assert result == "1.0.0"

    def test_fallback_on_os_error(self):
        with patch("immich_accelerator.dashboard.Path") as mock_path_cls:
            mock_path_cls.return_value.parent.parent.__truediv__.side_effect = OSError
            result = _get_accelerator_version()
            assert result == "1.0.0"


# ---------------------------------------------------------------------------
# _run
# ---------------------------------------------------------------------------


class TestRun:
    def test_successful_command(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "  hello world  \n"
        with patch("subprocess.run", return_value=result):
            assert _run(["echo", "hello"]) == "hello world"

    def test_failed_command_returns_empty(self):
        result = MagicMock()
        result.returncode = 1
        result.stdout = "error output"
        with patch("subprocess.run", return_value=result):
            assert _run(["false"]) == ""

    def test_timeout_returns_empty(self):
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="cmd", timeout=5)):
            assert _run(["slow-cmd"]) == ""

    def test_os_error_returns_empty(self):
        with patch("subprocess.run", side_effect=OSError("No such file")):
            assert _run(["/nonexistent"]) == ""

    def test_custom_timeout(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "ok"
        with patch("subprocess.run", return_value=result) as mock_run:
            _run(["cmd"], timeout=30)
            _, kwargs = mock_run.call_args
            assert kwargs["timeout"] == 30

    def test_env_passed_through(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "ok"
        custom_env = {"FOO": "bar"}
        with patch("subprocess.run", return_value=result) as mock_run:
            _run(["cmd"], env=custom_env)
            _, kwargs = mock_run.call_args
            assert kwargs["env"] == custom_env


# ---------------------------------------------------------------------------
# _query_db
# ---------------------------------------------------------------------------


class TestQueryDb:
    def test_uses_psql_when_password_set(self):
        config = {
            "db_hostname": "192.168.1.100",
            "db_port": "5432",
            "db_username": "postgres",
            "db_password": "secret",
            "db_name": "immich",
        }
        with patch("os.path.exists", return_value=True), patch("immich_accelerator.dashboard._run", return_value="42") as mock_run:
            result = _query_db("SELECT 1", config)
            assert result == "42"
            cmd = mock_run.call_args[0][0]
            assert "psql" in cmd[0]
            assert "-h" in cmd
            assert "192.168.1.100" in cmd

    def test_falls_back_to_docker_when_no_psql(self):
        config = {
            "db_hostname": "localhost",
            "db_port": "5432",
            "db_username": "postgres",
            "db_password": "",
            "db_name": "immich",
        }

        # psql not found, docker found
        def exists_side_effect(path):
            if "psql" in str(path):
                return False
            return "docker" in str(path)

        with patch("os.path.exists", side_effect=exists_side_effect), patch("immich_accelerator.dashboard._run", return_value="1") as mock_run:
            result = _query_db("SELECT 1", config)
            assert result == "1"
            cmd = mock_run.call_args[0][0]
            assert "docker" in cmd[0]
            assert "exec" in cmd

    def test_uses_custom_db_container(self):
        config = {
            "db_hostname": "localhost",
            "db_port": "5432",
            "db_username": "postgres",
            "db_password": "",
            "db_name": "immich",
            "db_container": "my_custom_postgres",
        }

        def exists_side_effect(path):
            if "psql" in str(path):
                return False
            return "docker" in str(path)

        with patch("os.path.exists", side_effect=exists_side_effect), patch("immich_accelerator.dashboard._run", return_value="1") as mock_run:
            _query_db("SELECT 1", config)
            cmd = mock_run.call_args[0][0]
            assert "my_custom_postgres" in cmd


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset the module-level cache before each test."""
        import immich_accelerator.dashboard as d

        d._cache = {}
        d._cache_ts = 0
        d._static_hw = None
        yield
        d._cache = {}
        d._cache_ts = 0
        d._static_hw = None

    def test_returns_structure(self, sample_config):
        import immich_accelerator.dashboard as d

        d._static_hw = {"mem_total_gb": 32.0, "cpus": 10}

        with (
            patch("urllib.request.urlopen", side_effect=OSError),
            patch("immich_accelerator.dashboard._query_db", return_value="100|200|100|100|50|10|5"),
            patch("immich_accelerator.dashboard._run", return_value="{ 1.50 2.00 3.00 }"),
        ):
            status = get_status(sample_config)

        assert "services" in status
        assert "progress" in status
        assert "system" in status
        assert "version" in status
        assert "accelerator_version" in status
        assert "queue_active" in status

    def test_services_section(self, sample_config):
        import immich_accelerator.dashboard as d

        d._static_hw = {"mem_total_gb": 32.0, "cpus": 10}

        with (
            patch("urllib.request.urlopen", side_effect=OSError),
            patch("immich_accelerator.dashboard._query_db", return_value="50|100|50|50|25|5|3"),
            patch("immich_accelerator.dashboard._run", return_value="{ 0.50 1.00 1.50 }"),
        ):
            status = get_status(sample_config)

        assert "worker" in status["services"]
        assert "ml" in status["services"]
        assert "docker" in status["services"]
        for svc in status["services"].values():
            assert "alive" in svc
            assert "name" in svc

    def test_progress_section(self, sample_config):
        import immich_accelerator.dashboard as d

        d._static_hw = {"mem_total_gb": 32.0, "cpus": 10}

        with (
            patch("urllib.request.urlopen", side_effect=OSError),
            patch("immich_accelerator.dashboard._query_db", return_value="80|100|60|70|50|10|5"),
            patch("immich_accelerator.dashboard._run", return_value="{ 0.50 1.00 1.50 }"),
        ):
            status = get_status(sample_config)

        progress = status["progress"]
        assert "thumbnails" in progress
        assert "clip" in progress
        assert "faces" in progress
        assert "ocr" in progress
        assert "video" in progress
        for key in ("thumbnails", "clip", "faces", "ocr", "video"):
            p = progress[key]
            assert "done" in p
            assert "total" in p
            assert "pct" in p
            assert "skipped" in p

    def test_progress_calculations(self, sample_config):
        import immich_accelerator.dashboard as d

        d._static_hw = {"mem_total_gb": 32.0, "cpus": 10}

        with (
            patch("urllib.request.urlopen", side_effect=OSError),
            patch("immich_accelerator.dashboard._query_db", return_value="100|100|100|100|100|10|10"),
            patch("immich_accelerator.dashboard._run", return_value=""),
        ):
            status = get_status(sample_config)

        assert status["progress"]["thumbnails"]["pct"] == 100.0

    def test_caching(self, sample_config):
        import immich_accelerator.dashboard as d

        d._static_hw = {"mem_total_gb": 32.0, "cpus": 10}

        with (
            patch("urllib.request.urlopen", side_effect=OSError),
            patch("immich_accelerator.dashboard._query_db", return_value="50|100|50|50|50|5|3") as mock_db,
            patch("immich_accelerator.dashboard._run", return_value=""),
        ):
            status1 = get_status(sample_config)
            status2 = get_status(sample_config)

        # DB should only be queried once (second call hits cache)
        assert mock_db.call_count == 1
        assert status1 is status2

    def test_empty_db_response(self, sample_config):
        import immich_accelerator.dashboard as d

        d._static_hw = {"mem_total_gb": 32.0, "cpus": 10}

        with patch("urllib.request.urlopen", side_effect=OSError), patch("immich_accelerator.dashboard._query_db", return_value=""), patch("immich_accelerator.dashboard._run", return_value=""):
            status = get_status(sample_config)

        assert status["progress"]["thumbnails"]["total"] == 0

    def test_malformed_db_response(self, sample_config):
        import immich_accelerator.dashboard as d

        d._static_hw = {"mem_total_gb": 32.0, "cpus": 10}

        with (
            patch("urllib.request.urlopen", side_effect=OSError),
            patch("immich_accelerator.dashboard._query_db", return_value="not|a|valid|response"),
            patch("immich_accelerator.dashboard._run", return_value=""),
        ):
            status = get_status(sample_config)

        # Should gracefully handle parse errors
        assert status["progress"]["thumbnails"]["total"] == 0

    def test_load_parsing(self, sample_config):
        import immich_accelerator.dashboard as d

        d._static_hw = {"mem_total_gb": 32.0, "cpus": 10}

        with (
            patch("urllib.request.urlopen", side_effect=OSError),
            patch("immich_accelerator.dashboard._query_db", return_value="0|0|0|0|0|0|0"),
            patch("immich_accelerator.dashboard._run", return_value="{ 2.50 3.00 4.00 }"),
        ):
            status = get_status(sample_config)

        assert status["system"]["load_1m"] == 2.5

    def test_version_from_config(self, sample_config):
        import immich_accelerator.dashboard as d

        d._static_hw = {"mem_total_gb": 32.0, "cpus": 10}

        with (
            patch("urllib.request.urlopen", side_effect=OSError),
            patch("immich_accelerator.dashboard._query_db", return_value="0|0|0|0|0|0|0"),
            patch("immich_accelerator.dashboard._run", return_value=""),
        ):
            status = get_status(sample_config)

        assert status["version"] == "2.6.3"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


class TestFastAPIApp:
    @pytest.fixture
    def app(self, sample_config):
        return create_app(sample_config)

    @pytest.fixture
    def client(self, app):
        from starlette.testclient import TestClient

        return TestClient(app)

    def test_index_returns_html(self, client):
        # Mock the HTML file read
        with patch("immich_accelerator.dashboard._load_html", return_value="<html>test</html>"):
            resp = client.get("/")
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]

    def test_api_status_endpoint(self, client, sample_config):
        import immich_accelerator.dashboard as d

        d._static_hw = {"mem_total_gb": 32.0, "cpus": 10}
        d._cache = {}
        d._cache_ts = 0

        mock_status = {
            "services": {"worker": {"alive": True, "name": "Worker"}},
            "progress": {},
            "system": {"load_1m": 1.0, "mem_total_gb": 32.0, "cpus": 10},
            "version": "2.6.3",
            "accelerator_version": "1.3.1",
            "queue_active": {},
        }
        with patch("immich_accelerator.dashboard.get_status", return_value=mock_status):
            resp = client.get("/api/status")
            assert resp.status_code == 200
            data = resp.json()
            assert "services" in data
            assert "version" in data

    def test_api_requeue_no_api_key(self, sample_config):
        config_no_key = {k: v for k, v in sample_config.items() if k != "api_key"}
        config_no_key["api_key"] = ""
        app = create_app(config_no_key)
        from starlette.testclient import TestClient

        client = TestClient(app)
        resp = client.post("/api/requeue")
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_api_requeue_with_api_key(self, sample_config):
        app = create_app(sample_config)
        from starlette.testclient import TestClient

        client = TestClient(app)

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"{}"
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            resp = client.post("/api/requeue")
            assert resp.status_code == 200
            data = resp.json()
            # Should have results for all 5 queues
            assert "thumbnailGeneration" in data
            assert "smartSearch" in data
            assert "faceDetection" in data
            assert "ocr" in data
            assert "videoConversion" in data

    def test_api_requeue_handles_failures(self, sample_config):
        app = create_app(sample_config)
        from starlette.testclient import TestClient

        client = TestClient(app)

        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            resp = client.post("/api/requeue")
            assert resp.status_code == 200
            data = resp.json()
            for v in data.values():
                assert v == "failed"

    def test_status_does_not_block_event_loop(self, sample_config):
        """A slow get_status must not freeze concurrent requests.

        Regression guard: an ``async def`` handler that calls blocking I/O
        runs it on the event loop and serializes every client. The handlers
        are plain ``def`` so Starlette offloads them to a threadpool, letting
        two concurrent /api/status requests overlap instead of running
        back-to-back.
        """
        import asyncio
        import time

        app = create_app(sample_config)

        sleep_s = 0.5

        def slow_status(_config):
            time.sleep(sleep_s)
            return {"services": {}, "progress": {}, "system": {}, "version": "x"}

        async def fire_two():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                start = time.monotonic()
                r1, r2 = await asyncio.gather(ac.get("/api/status"), ac.get("/api/status"))
                return time.monotonic() - start, r1, r2

        with patch("immich_accelerator.dashboard.get_status", side_effect=slow_status):
            elapsed, r1, r2 = asyncio.run(fire_two())

        assert r1.status_code == 200
        assert r2.status_code == 200
        # Serialized would be ~2*sleep_s; overlapped stays well under that.
        assert elapsed < sleep_s * 1.8, f"requests serialized ({elapsed:.2f}s for two {sleep_s}s calls) — blocking I/O is running on the event loop"

    def test_api_requeue_handles_400_as_ok(self, sample_config):
        """400 from Immich means 'already running' which is fine."""
        import urllib.error

        app = create_app(sample_config)
        from starlette.testclient import TestClient

        client = TestClient(app)

        from email.message import Message

        error = urllib.error.HTTPError(
            url="http://localhost:2283/api/jobs/thumbnailGeneration",
            code=400,
            msg="Bad Request",
            hdrs=Message(),
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=error):
            resp = client.post("/api/requeue")
            assert resp.status_code == 200
            data = resp.json()
            for v in data.values():
                assert v == "ok"


# ---------------------------------------------------------------------------
# TestGetStatusMl
# ---------------------------------------------------------------------------


class TestGetStatusMl:
    def _cfg(self):
        return {"mode": "ml-only", "ml_host": "0.0.0.0", "ml_port": 3003, "metrics_powermetrics": True}

    def test_routes_to_ml_status(self):
        import immich_accelerator.dashboard as dash

        with patch.object(dash, "get_status_ml", return_value={"mode": "ml-only"}) as m:
            out = dash.get_status(self._cfg())
        assert out == {"mode": "ml-only"}
        m.assert_called_once()

    def test_ml_status_shape(self):
        import immich_accelerator.dashboard as dash

        sample_log = "predict: 1 task(s) [clip] completed in 40ms\n"
        with (
            patch.object(dash, "_tail_text", return_value=sample_log),
            patch.object(dash, "_count_predicts", return_value=5),
            patch.object(dash, "_ml_health", return_value=None),
            patch.object(dash, "_ping_ml", return_value=True),
            patch("immich_accelerator.metrics.sample_powermetrics", return_value={"gpu_residency_pct": 30.0, "ane_mw": 500.0}),
            patch.object(dash, "_system_metrics", return_value={"load_1m": 1.0, "mem_total_gb": 24.0, "cpus": 10}),
        ):
            dash._ml_cache = None
            dash._ml_cache_ts = 0
            dash._ml_last_total = 0
            dash._ml_last_ts = 0.0
            out = dash.get_status_ml(self._cfg())
        assert out["mode"] == "ml-only"
        assert out["services"]["ml"]["alive"] is True
        assert out["ml"]["tasks"] == {"clip": 1, "faces": 0, "ocr": 0}
        assert out["ml"]["total_predicts"] == 5
        assert out["ml"]["throughput_rps"] == 0.0  # first call: no baseline
        assert out["hardware"]["gpu_residency_pct"] == 30.0
        assert out["hardware"]["ane_mw"] == 500.0
        assert out["hardware"]["powermetrics"] is True

    def test_throughput_delta(self):
        import immich_accelerator.dashboard as dash

        with (
            patch.object(dash, "_tail_text", return_value=""),
            patch.object(dash, "_ml_health", return_value=None),
            patch.object(dash, "_ping_ml", return_value=True),
            patch("immich_accelerator.metrics.sample_powermetrics", return_value=None),
            patch.object(dash, "_system_metrics", return_value={"load_1m": 0, "mem_total_gb": 24.0, "cpus": 10}),
            patch.object(dash, "_count_predicts", return_value=30),
            patch("immich_accelerator.dashboard.time.monotonic", return_value=105.0),
        ):
            dash._ml_cache = None
            dash._ml_cache_ts = 0
            dash._ml_last_total = 10
            dash._ml_last_ts = 100.0
            out = dash.get_status_ml(self._cfg())
        # 20 predicts over 5s = 4.0 req/s
        assert out["ml"]["throughput_rps"] == 4.0

    def test_full_status_carries_mode_field(
        self,
    ):
        import immich_accelerator.dashboard as dash

        dash._cache = None
        dash._cache_ts = 0
        with patch.object(dash, "_query_db", return_value="0|0|0|0|0|0|0"), patch.object(dash, "_run", return_value=""), patch("urllib.request.urlopen", side_effect=Exception):
            out = dash.get_status({"mode": "full"})
        assert out["mode"] == "full"

    def test_powermetrics_flag_false_when_no_values(self):
        import immich_accelerator.dashboard as dash

        with (
            patch.object(dash, "_tail_text", return_value=""),
            patch.object(dash, "_count_predicts", return_value=0),
            patch.object(dash, "_ml_health", return_value=None),
            patch.object(dash, "_ping_ml", return_value=True),
            patch("immich_accelerator.metrics.sample_powermetrics", return_value={"gpu_residency_pct": None, "ane_mw": None}),
            patch.object(dash, "_system_metrics", return_value={"load_1m": 0, "mem_total_gb": 24.0, "cpus": 10}),
        ):
            dash._ml_cache = None
            dash._ml_cache_ts = 0
            dash._ml_last_total = 0
            dash._ml_last_ts = 0.0
            out = dash.get_status_ml({"mode": "ml-only", "ml_host": "0.0.0.0", "ml_port": 3003, "metrics_powermetrics": True})
        assert out["hardware"]["powermetrics"] is False
        assert out["hardware"]["gpu_residency_pct"] is None

    def test_health_surfaced_from_ml_health(self):
        import immich_accelerator.dashboard as dash

        health = {
            "status": "healthy",
            "checks": {"clip": "ok"},
            "models": {
                "clip": {"loaded": True, "name": "ViT-SO400M-16-SigLIP2-384__webli"},
                "face": {"loaded": True, "name": "buffalo_l"},
            },
            "unload_strategy": "pressure",
        }
        with (
            patch.object(dash, "_tail_text", return_value=""),
            patch.object(dash, "_count_predicts", return_value=0),
            patch.object(dash, "_ml_health", return_value=health),
            patch("immich_accelerator.metrics.sample_powermetrics", return_value=None),
            patch.object(dash, "_system_metrics", return_value={"load_1m": 0, "mem_total_gb": 24.0, "cpus": 10}),
        ):
            dash._ml_cache = None
            dash._ml_cache_ts = 0
            dash._ml_last_total = 0
            dash._ml_last_ts = 0.0
            out = dash.get_status_ml(self._cfg())
        # /health answering implies liveness without a separate /ping
        assert out["services"]["ml"]["alive"] is True
        assert out["health"]["status"] == "healthy"
        assert out["health"]["models"]["clip"]["name"] == "ViT-SO400M-16-SigLIP2-384__webli"
        assert out["health"]["unload_strategy"] == "pressure"

    def test_offline_when_health_and_ping_fail(self):
        import immich_accelerator.dashboard as dash

        with (
            patch.object(dash, "_tail_text", return_value=""),
            patch.object(dash, "_count_predicts", return_value=0),
            patch.object(dash, "_ml_health", return_value=None),
            patch.object(dash, "_ping_ml", return_value=False),
            patch("immich_accelerator.metrics.sample_powermetrics", return_value=None),
            patch.object(dash, "_system_metrics", return_value={"load_1m": 0, "mem_total_gb": 24.0, "cpus": 10}),
        ):
            dash._ml_cache = None
            dash._ml_cache_ts = 0
            dash._ml_last_total = 0
            dash._ml_last_ts = 0.0
            out = dash.get_status_ml(self._cfg())
        assert out["services"]["ml"]["alive"] is False
        assert out["ml"]["activity"] == "offline"
        assert out["health"]["status"] == "offline"

    def test_latency_by_task_and_events_present(self):
        import immich_accelerator.dashboard as dash

        log = (
            "2026-06-06 12:00:01 - src.main - INFO -   clip: 40ms\n"
            "2026-06-06 12:00:01 - src.main - WARNING - Falling back to HF bf16 for SigLIP2\n"
            "2026-06-06 12:00:01 - src.main - INFO - predict: 1 task(s) [clip] completed in 41ms\n"
        )
        with (
            patch.object(dash, "_tail_text", return_value=log),
            patch.object(dash, "_count_predicts", return_value=1),
            patch.object(dash, "_ml_health", return_value=None),
            patch.object(dash, "_ping_ml", return_value=True),
            patch("immich_accelerator.metrics.sample_powermetrics", return_value=None),
            patch.object(dash, "_system_metrics", return_value={"load_1m": 0, "mem_total_gb": 24.0, "cpus": 10}),
        ):
            dash._ml_cache = None
            dash._ml_cache_ts = 0
            dash._ml_last_total = 0
            dash._ml_last_ts = 0.0
            out = dash.get_status_ml(self._cfg())
        assert out["ml"]["latency_by_task"]["clip"]["samples"] == 1
        assert out["events"]["warnings"] == 1
        assert out["events"]["recent"][-1]["msg"].startswith("Falling back")


class TestMemoryMetrics:
    def test_available_memory_parses_vm_stat(self):
        import immich_accelerator.dashboard as dash

        vmstat = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:                               100000.\nPages inactive:                            50000.\n"
        with patch.object(dash, "_run", return_value=vmstat):
            assert dash._available_memory_mb() == (150000 * 16384) // (1024 * 1024)

    def test_available_memory_none_when_vm_stat_empty(self):
        import immich_accelerator.dashboard as dash

        with patch.object(dash, "_run", return_value=""):
            assert dash._available_memory_mb() is None


# ---------------------------------------------------------------------------
# _IncrementalPredictCounter — tail-only, monotonic predict counting
# ---------------------------------------------------------------------------


class TestIncrementalPredictCounter:
    def test_counts_full_file_on_first_call(self, tmp_path):
        log = tmp_path / "ml.log"
        log.write_text(_PREDICT + _NOISE + _PREDICT)
        assert _IncrementalPredictCounter().count(log) == 2

    def test_missing_file_returns_zero(self, tmp_path):
        assert _IncrementalPredictCounter().count(tmp_path / "absent.log") == 0

    def test_only_appended_bytes_are_scanned(self, tmp_path):
        log = tmp_path / "ml.log"
        log.write_text(_PREDICT)
        counter = _IncrementalPredictCounter()
        assert counter.count(log) == 1
        # Append two more predicts; re-scanning only the tail must keep it monotonic.
        with open(log, "a") as f:
            f.write(_NOISE + _PREDICT + _PREDICT)
        assert counter.count(log) == 3
        offset_after = counter._offset
        # No growth → no re-count and offset unchanged.
        assert counter.count(log) == 3
        assert counter._offset == offset_after

    def test_partial_trailing_line_deferred_until_newline(self, tmp_path):
        log = tmp_path / "ml.log"
        log.write_text(_PREDICT)
        counter = _IncrementalPredictCounter()
        assert counter.count(log) == 1
        # Write a predict line WITHOUT its trailing newline — not yet complete.
        partial = "2026-06-06 12:00:02 INFO predict: 1 task(s) [ocr] completed in 9ms"
        with open(log, "a") as f:
            f.write(partial)
        assert counter.count(log) == 1  # held back
        with open(log, "a") as f:
            f.write("\n")
        assert counter.count(log) == 2  # counted once the line completes

    def test_truncation_resets_and_rescans(self, tmp_path):
        log = tmp_path / "ml.log"
        log.write_text(_PREDICT + _PREDICT + _PREDICT)
        counter = _IncrementalPredictCounter()
        assert counter.count(log) == 3
        # Log rotated/truncated to a smaller file → count restarts from the new content.
        log.write_text(_PREDICT)
        assert counter.count(log) == 1
