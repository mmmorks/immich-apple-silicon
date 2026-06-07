"""Immich Accelerator Dashboard — real-time monitoring web UI.

A lightweight FastAPI server that exposes the accelerator's status as
both API endpoints and a beautiful single-page dashboard. Polls the
Immich database, checks service health, and reads system metrics.

Usage:
    python -m immich_accelerator dashboard          # http://localhost:8422
    python -m immich_accelerator dashboard --port 9000

Security note: The dashboard renders data from the local Immich database
and system metrics. All data sources are trusted (localhost only). The
HTML rendering uses template literals with numeric/string data from our
own API — no user-supplied content is rendered as HTML.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("dashboard")

# Cache to avoid hammering the DB on every request
_cache: dict = {}
_cache_ts: float = 0
_CACHE_TTL = 3  # seconds

_static_hw: dict | None = None

# ML appliance mode cache
_ml_cache = None
_ml_cache_ts = 0.0
_ml_last_total = 0
_ml_last_ts = 0.0

# Serializes the cache check-and-update in get_status / get_status_ml. The
# handlers run in a threadpool (sync defs), so concurrent requests would
# otherwise race the module-global caches and the throughput rate's
# read-modify-write. Held across the body, it also gives single-flight caching.
_status_lock = threading.Lock()


def _get_accelerator_version() -> str:
    """Get accelerator version from the VERSION file or fall back."""
    try:
        version_file = Path(__file__).parent.parent / "VERSION"
        if version_file.exists():
            return version_file.read_text().strip()
    except OSError:
        pass
    return "1.0.0"


def _run(cmd: list[str], timeout: int = 5, env: dict | None = None) -> str:
    """Run a command and return stdout, or empty string on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (subprocess.SubprocessError, OSError):
        return ""


_db_error_logged = False


def _query_db(sql: str, config: dict) -> str:
    """Run a SQL query against Immich's Postgres.

    Uses direct psql connection when DB host/password are configured (remote
    setups). Falls back to docker exec for local setups (backwards compat).
    """
    global _db_error_logged
    host = config.get("db_hostname", "localhost")
    port = config.get("db_port", "5432")
    user = config.get("db_username", "postgres")
    password = config.get("db_password", "")
    db = config.get("db_name", "immich")

    # Direct psql connection — works for both local and remote setups
    psql = "/opt/homebrew/opt/libpq/bin/psql"
    if not os.path.exists(psql):
        psql = "/opt/homebrew/bin/psql"
    if not os.path.exists(psql):
        psql = "/usr/local/bin/psql"

    has_psql = os.path.exists(psql)

    # Try direct psql connection (remote setups, or local with password)
    if has_psql and (password or host != "localhost"):
        env = {**os.environ}
        if password:
            env["PGPASSWORD"] = password
        result = _run(
            [psql, "-h", host, "-p", port, "-U", user, "-d", db, "-t", "-A", "-c", sql],
            env=env,
        )
        if result:
            _db_error_logged = False
            return result
        # Don't return — fall through to docker exec fallback

    # Fallback: docker exec (local setups, or psql failed above)
    docker = "/usr/local/bin/docker"
    if not os.path.exists(docker):
        docker = "/opt/homebrew/bin/docker"
    if os.path.exists(docker):
        container = config.get("db_container", "immich_postgres")
        result = _run(
            [
                docker,
                "exec",
                container,
                "psql",
                "-U",
                user,
                "-d",
                db,
                "-t",
                "-A",
                "-c",
                sql,
            ]
        )
        if result:
            _db_error_logged = False
            return result

    # Nothing worked — log once
    if not _db_error_logged:
        if not has_psql:
            log.warning("Dashboard: psql not found. Install with: brew install libpq")
        elif host != "localhost":
            log.warning("Dashboard: cannot reach Postgres at %s:%s", host, port)
            log.warning("  Check that the port is exposed (not 127.0.0.1) and reachable from this Mac")
        else:
            log.warning("Dashboard: cannot connect to Postgres. Check that Docker is running.")
        _db_error_logged = True
    return ""


def get_status(config: dict) -> dict:
    """Get full accelerator status. Cached for _CACHE_TTL seconds."""
    if config.get("mode") == "ml-only":
        return get_status_ml(config)

    global _cache, _cache_ts
    with _status_lock:
        now = time.monotonic()
        if now - _cache_ts < _CACHE_TTL and _cache:
            return _cache

        # Service health
        import urllib.request as _urlreq

        ml_alive = False
        try:
            with _urlreq.urlopen("http://localhost:3003/ping", timeout=2) as r:
                ml_alive = r.read().decode().strip() == "pong"
        except Exception:
            pass

        # Check worker PID file (more reliable than pgrep — process name is 'node', not 'immich')
        worker_alive = False
        worker_rss_mb = 0
        pid_file = Path.home() / ".immich-accelerator" / "pids" / "worker.pid"
        try:
            if pid_file.exists():
                pid = int(pid_file.read_text().strip().split("\n")[0])
                os.kill(pid, 0)  # check if process exists
                worker_alive = True
                # Grab RSS for memory-growth detection. On macOS `ps -o rss=`
                # returns kilobytes. Rising RSS over hours suggests a libvips
                # or Sharp memory leak causing the thumbnail slowdown (#33).
                rss_out = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "rss="],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if rss_out.returncode == 0 and rss_out.stdout.strip():
                    worker_rss_mb = round(int(rss_out.stdout.strip()) / 1024)
        except (ValueError, OSError, subprocess.SubprocessError):
            pass

        # Processing counts. The SELECT emits seven aggregates as one
        # pipe-delimited row, unpacked positionally below. The column order
        # here and the unpack order MUST stay in lockstep — reordering one
        # without the other silently mis-assigns counts. Positions (0-based):
        #   0 thumbs   1 total   2 clip   3 faces
        #   4 ocr      5 total_videos     6 encoded_videos
        # Exclude hidden assets (Live Photo motion files) — Immich skips them too
        counts_raw = _query_db(
            "SELECT COUNT(*) FILTER (WHERE thumbhash IS NOT NULL), COUNT(*), "  # 0 thumbs, 1 total
            "(SELECT COUNT(*) FROM smart_search), "  # 2 clip
            '(SELECT COUNT(*) FROM asset_job_status WHERE "facesRecognizedAt" IS NOT NULL), '  # 3 faces
            '(SELECT COUNT(*) FROM asset_job_status WHERE "ocrAt" IS NOT NULL), '  # 4 ocr
            "COUNT(*) FILTER (WHERE type = 'VIDEO' AND visibility != 'hidden'), "  # 5 total_videos
            "(SELECT COUNT(*) FROM asset_file af JOIN asset a ON a.id = af.\"assetId\" WHERE af.type = 'encoded_video' AND a.visibility != 'hidden') "  # 6 encoded_videos
            "FROM asset WHERE \"deletedAt\" IS NULL AND visibility != 'hidden'",
            config,
        )

        thumbs, total, clip, faces, ocr, total_videos, encoded_videos = 0, 0, 0, 0, 0, 0, 0
        if counts_raw and "|" in counts_raw:
            parts = counts_raw.split("|")
            if len(parts) == 7:
                with contextlib.suppress(ValueError):
                    # Order mirrors the SELECT column order documented above.
                    thumbs, total, clip, faces, ocr, total_videos, encoded_videos = [int(p) for p in parts]

        # System metrics
        load_raw = _run(["sysctl", "-n", "vm.loadavg"])
        load_1m = 0.0
        if load_raw:
            with contextlib.suppress(ValueError, IndexError):
                load_1m = float(load_raw.strip("{ }").split()[0])

        # Static hardware info (never changes, cached on first call)
        global _static_hw
        if _static_hw is None:
            mem_raw = _run(["sysctl", "-n", "hw.memsize"])
            cpu_raw = _run(["sysctl", "-n", "hw.ncpu"])
            _static_hw = {
                "mem_total_gb": round(int(mem_raw) / (1024**3), 1) if mem_raw else 0,
                "cpus": int(cpu_raw) if cpu_raw else 0,
            }

        # Per-queue activity from Immich jobs API. Also capture the raw
        # active + waiting counts so the frontend can show "X remaining"
        # (matching what the Immich admin panel shows) instead of only
        # displaying DB-derived done/total which measures a different thing.
        queue_status = {}
        queue_counts = {}
        api_key = config.get("api_key", "")
        immich_url = config.get("immich_url", "http://localhost:2283")
        jobs_api_error = ""
        if api_key:
            import urllib.request as _urlreq2

            try:
                req = _urlreq2.Request(f"{immich_url}/api/jobs", headers={"x-api-key": api_key})
                with _urlreq2.urlopen(req, timeout=5) as r:
                    body = r.read()
                    if not body or not body.strip():
                        raise ValueError(f"empty response from {immich_url}/api/jobs")
                    jobs = json.loads(body)
                    queue_map = {
                        "thumbnailGeneration": "thumbnails",
                        "smartSearch": "clip",
                        "faceDetection": "faces",
                        "ocr": "ocr",
                        "videoConversion": "video",
                    }
                    for immich_name, our_name in queue_map.items():
                        counts = jobs.get(immich_name, {}).get("jobCounts", {})
                        active = counts.get("active", 0)
                        waiting = counts.get("waiting", 0)
                        queue_status[our_name] = (active + waiting) > 0
                        queue_counts[our_name] = active + waiting
            except Exception as e:
                err = str(e)
                # Make common errors human-readable
                if "Expecting value" in err or "empty response" in err:
                    jobs_api_error = "Immich API returned empty response (check immich_url in config)"
                elif "401" in err or "403" in err:
                    jobs_api_error = "API key rejected (check api_key in config)"
                elif "Connection refused" in err or "ECONNREFUSED" in err:
                    jobs_api_error = f"cannot reach {immich_url} (is Immich running?)"
                elif "timed out" in err.lower():
                    jobs_api_error = "Immich API timed out (server under heavy load?)"
                else:
                    jobs_api_error = err[:200]
                log.warning("jobs API unreachable: %s", jobs_api_error)
        else:
            jobs_api_error = "no api_key configured"

        # Versions
        version = config.get("version", "?")

        # When all queues are confirmed idle (API responded, nothing active),
        # unprocessable assets are "skipped." Only apply when we actually got
        # queue data — empty queue_status means API unreachable, not "idle."
        queues_known = bool(queue_status)
        any_active = queues_known and any(queue_status.values())

        def prog(done, tot):
            if queues_known and not any_active and done < tot:
                return {"done": done, "total": tot, "pct": 100.0, "skipped": tot - done}
            return {
                "done": done,
                "total": tot,
                "pct": round(done / max(tot, 1) * 100, 1),
                "skipped": 0,
            }

        # Video transcode: use queue state for pct when active, 100% when idle + transcoded
        vid_active = queue_status.get("video", False)
        if vid_active and total_videos > 0:
            vid_pct = round(encoded_videos / total_videos * 100, 1)
        elif encoded_videos > 0:
            vid_pct = 100.0
        else:
            vid_pct = 0

        status = {
            "mode": "full",
            "services": {
                "worker": {
                    "alive": worker_alive,
                    "name": "Microservices Worker",
                    "rss_mb": worker_rss_mb,
                },
                "ml": {"alive": ml_alive, "name": "ML Service"},
                "docker": {"alive": total > 0, "name": "Docker (API)"},
            },
            "progress": {
                "thumbnails": prog(thumbs, total),
                "clip": prog(clip, total),
                "faces": prog(faces, total),
                "ocr": prog(ocr, total),
                "video": {
                    "done": encoded_videos,
                    "total": total_videos,
                    "pct": vid_pct,
                    "skipped": 0,
                },
            },
            "system": {
                "load_1m": load_1m,
                "mem_total_gb": _static_hw["mem_total_gb"],
                "cpus": _static_hw["cpus"],
            },
            "version": version,
            "accelerator_version": _get_accelerator_version(),
            "queue_active": queue_status,
            "queue_counts": queue_counts,
            "jobs_api_error": jobs_api_error,
        }

        _cache = status
        _cache_ts = now
        return status


def _ping_ml(config: dict) -> bool:
    import urllib.request as _urlreq

    port = int(config.get("ml_port", 3003))
    try:
        with _urlreq.urlopen(f"http://localhost:{port}/ping", timeout=2) as r:
            return r.read().decode().strip() == "pong"
    except Exception:
        return False


def _ml_health(config: dict) -> dict | None:
    """Fetch the ML service's /health JSON (model load state, degraded status).

    Returns the parsed dict, or None if unreachable / not valid JSON. Short
    timeout; never raises. /health is the fork's own endpoint (upstream Immich
    ML has none), so depending on it here is safe.
    """
    import urllib.request as _urlreq

    port = int(config.get("ml_port", 3003))
    try:
        with _urlreq.urlopen(f"http://localhost:{port}/health", timeout=3) as r:
            data = json.loads(r.read())
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _log_age_seconds(path: Path) -> float | None:
    """Seconds since the ML log was last written — a proxy for recent activity.

    Distinguishes a quiet-but-healthy service (old log, /ping ok) from a busy
    one (fresh log). Returns None if the log doesn't exist yet.
    """
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _tail_text(path: Path, max_bytes: int = 65536) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


class _IncrementalPredictCounter:
    """Cumulative count of predict-completion lines in a growing log.

    A naive full-file scan on every dashboard cache-miss is O(file size) and
    grows unbounded as ml.log accumulates. Instead we remember a byte offset
    (always at a line boundary) and a running count, and on each call scan
    only the bytes appended since last time. The first call reads the whole
    file once to establish the baseline; subsequent calls read just the tail.

    Truncation/rotation is detected by the file shrinking below our offset, in
    which case we reset and rescan from the start. A trailing partial line (no
    newline yet) is left uncounted until its newline arrives, so a line being
    written mid-scan is never split or double-counted.
    """

    def __init__(self) -> None:
        self._offset = 0  # bytes counted so far (ends at a newline)
        self._count = 0  # running predict-line total

    def count(self, path: Path) -> int:
        from .ml_stats import count_predict_lines

        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size < self._offset:  # truncated/rotated → start over
                    self._offset = 0
                    self._count = 0
                f.seek(self._offset)
                chunk = f.read(size - self._offset)
        except OSError:
            return self._count

        last_nl = chunk.rfind(b"\n")
        if last_nl == -1:
            return self._count  # no complete line appended yet
        complete = chunk[: last_nl + 1].decode("utf-8", "replace")
        self._count += count_predict_lines(complete.splitlines())
        self._offset += last_nl + 1
        return self._count


_predict_counter = _IncrementalPredictCounter()


def _count_predicts(path: Path) -> int:
    """Cumulative, monotonic count of predict-completion lines in the log.

    Scans only the bytes appended since the last call (see
    ``_IncrementalPredictCounter``); the windowed ``parse_ml_log`` total is
    not cumulative.
    """
    return _predict_counter.count(path)


def _available_memory_mb() -> int | None:
    """Available memory in MB (free + inactive pages) via vm_stat, or None.

    Mirrors the ML service's own memory check — this is the resource that drives
    the model unload strategy, so surfacing it predicts unload thrash.
    """
    import re

    out = _run(["vm_stat"])
    if not out:
        return None
    ps = re.search(r"page size of (\d+) bytes", out)
    page = int(ps.group(1)) if ps else 16384
    free = re.search(r"Pages free:\s+(\d+)", out)
    inactive = re.search(r"Pages inactive:\s+(\d+)", out)
    if not free or not inactive:
        return None
    return (int(free.group(1)) + int(inactive.group(1))) * page // (1024 * 1024)


def _system_metrics() -> dict:
    global _static_hw
    load_raw = _run(["sysctl", "-n", "vm.loadavg"])
    load_1m = 0.0
    if load_raw:
        with contextlib.suppress(ValueError, IndexError):
            load_1m = float(load_raw.strip("{ }").split()[0])
    if _static_hw is None:
        mem_raw = _run(["sysctl", "-n", "hw.memsize"])
        cpu_raw = _run(["sysctl", "-n", "hw.ncpu"])
        _static_hw = {
            "mem_total_gb": round(int(mem_raw) / (1024**3), 1) if mem_raw else 0,
            "cpus": int(cpu_raw) if cpu_raw else 0,
        }
    metrics = {"load_1m": load_1m, "mem_total_gb": _static_hw["mem_total_gb"], "cpus": _static_hw["cpus"]}
    # Available memory + used % (the unload-strategy-relevant numbers), when vm_stat is readable.
    avail_mb = _available_memory_mb()
    total_gb = _static_hw["mem_total_gb"]
    if avail_mb is not None and total_gb:
        avail_gb = round(avail_mb / 1024, 1)
        metrics["mem_available_gb"] = avail_gb
        metrics["mem_used_pct"] = round(max(0.0, min(1.0, (total_gb - avail_gb) / total_gb)) * 100, 1)
    return metrics


def get_status_ml(config: dict) -> dict:
    """ML appliance status: health, throughput, latency, real GPU/ANE."""
    global _ml_cache, _ml_cache_ts, _ml_last_total, _ml_last_ts
    from . import metrics
    from .ml_stats import parse_ml_events, parse_ml_log

    with _status_lock:
        now = time.monotonic()
        if _ml_cache and now - _ml_cache_ts < _CACHE_TTL:
            return _ml_cache

        log_path = Path.home() / ".immich-accelerator" / "logs" / "ml.log"
        tail = _tail_text(log_path)  # recent tail: tasks, latency, events
        stats = parse_ml_log(tail)
        events = parse_ml_events(tail)
        cumulative = _count_predicts(log_path)  # monotonic full-file count

        # /health gives model load state + degraded status, and implies liveness,
        # so only fall back to a bare /ping if it didn't answer.
        health = _ml_health(config)
        ml_alive = health is not None or _ping_ml(config)

        rate = 0.0
        if _ml_last_ts and now > _ml_last_ts and cumulative >= _ml_last_total:
            rate = (cumulative - _ml_last_total) / (now - _ml_last_ts)
        _ml_last_total = cumulative
        _ml_last_ts = now

        # Distinguish "active" (fresh log) from "idle" (responding but quiet) and
        # "offline" (not responding) — all three otherwise read as 0 req/s.
        log_age = _log_age_seconds(log_path)
        if not ml_alive:
            activity = "offline"
        elif log_age is not None and log_age < 15:
            activity = "active"
        else:
            activity = "idle"

        pm = metrics.sample_powermetrics() if config.get("metrics_powermetrics") else None

        status = {
            "mode": "ml-only",
            "services": {"ml": {"alive": ml_alive, "name": "ML Service"}},
            "ml": {
                "throughput_rps": round(rate, 2),
                "tasks": stats["tasks"],
                "latency_ms": stats["latency_ms"],
                "latency_by_task": stats["latency_by_task"],
                "total_predicts": cumulative,
                "endpoint": f"http://{config.get('ml_host', '0.0.0.0')}:{config.get('ml_port', 3003)}",
                "activity": activity,
                "last_activity_age_s": round(log_age, 1) if log_age is not None else None,
            },
            "health": {
                "status": (health or {}).get("status", "unknown" if ml_alive else "offline"),
                "checks": (health or {}).get("checks", {}),
                "models": (health or {}).get("models"),
                "unload_strategy": (health or {}).get("unload_strategy"),
            },
            "events": events,
            "hardware": {
                "gpu_residency_pct": pm.get("gpu_residency_pct") if pm else None,
                "ane_mw": pm.get("ane_mw") if pm else None,
                "powermetrics": bool(pm and (pm.get("gpu_residency_pct") is not None or pm.get("ane_mw") is not None)),
            },
            "system": _system_metrics(),
            "version": config.get("version", "—"),
            "accelerator_version": _get_accelerator_version(),
        }
        _ml_cache = status
        _ml_cache_ts = now
        return status


def _load_html() -> str:
    """Load the dashboard HTML from the static file."""
    html_path = Path(__file__).parent / "dashboard.html"
    return html_path.read_text()


def create_app(config: dict):
    """Create the FastAPI dashboard app."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Immich Accelerator Dashboard")

    # Handlers are sync defs on purpose: Starlette offloads sync path operations
    # to a threadpool, so their blocking file/subprocess/HTTP work never runs on
    # the async event loop and one slow downstream call can't freeze other clients.

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _load_html()

    @app.get("/api/status")
    def api_status():
        return JSONResponse(get_status(config))

    @app.post("/api/requeue")
    def api_requeue():
        """Trigger 'Run All Missing' for thumbnail, CLIP, faces, and OCR queues."""
        import urllib.error
        import urllib.request

        if config.get("mode") == "ml-only":
            return JSONResponse({"error": "requeue disabled in ml-only mode"}, status_code=400)

        api_key = config.get("api_key", "")
        immich_url = config.get("immich_url", "http://localhost:2283")
        if not api_key:
            return JSONResponse({"error": "No API key configured"}, status_code=400)

        results = {}
        for queue in [
            "thumbnailGeneration",
            "smartSearch",
            "faceDetection",
            "ocr",
            "videoConversion",
        ]:
            try:
                data = b'{"command": "start", "force": false}'
                req = urllib.request.Request(
                    f"{immich_url}/api/jobs/{queue}",
                    data=data,
                    method="PUT",
                    headers={
                        "x-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=10):
                    results[queue] = "ok"
            except urllib.error.HTTPError as e:
                # 400 "already running" is fine — job was already queued
                results[queue] = "ok" if e.code == 400 else "failed"
            except Exception:
                results[queue] = "failed"

        return JSONResponse(results)

    return app


def run_dashboard(config: dict, port: int = 8420):
    """Start the dashboard server."""
    import uvicorn

    app = create_app(config)
    log.info("Dashboard: http://localhost:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
