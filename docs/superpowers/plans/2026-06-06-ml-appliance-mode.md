# ML Appliance Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class `ml-only` ("ML appliance") mode that runs the native Metal/ANE ML service as a standalone remote ML endpoint for a separate Immich host, with no Docker/worker/DB/shared-filesystem on the Mac.

**Architecture:** A `mode` field in config (`"full"` default, `"ml-only"` new). Each `cmd_*` in `immich_accelerator/__main__.py` branches early on `mode`; ml-only paths skip all worker/Docker/path logic and only manage the ML service (+ dashboard). Two new pure leaf modules (`metrics.py`, `ml_stats.py`) provide testable parsers for `powermetrics` output and the ML request log. The dashboard gains an ML-focused status path and frontend layout.

**Tech Stack:** Python 3.11+ (stdlib + FastAPI/uvicorn for dashboard), pytest, macOS `powermetrics`/`sudo`/`visudo`, the `immich-ml-metal` submodule (`python -m src.main`, env `ML_HOST`/`ML_PORT`).

**Conventions for this plan:**
- Work happens on `main` of the fork (`origin` = `mmmorks/immich-apple-silicon`). No PRs.
- Every commit message ends with this trailer (shown once; include it in every commit):
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```
- Run tests with `python -m pytest` (never pipe into `tail`/`head` without `set -o pipefail`).
- TDD: write the failing test, see it fail, implement, see it pass, commit.

**Verified facts the code depends on:**
- ML service routes: `GET /ping` → `pong`, `GET /health`, `POST /predict`. Binds `ML_HOST` (default `0.0.0.0`) `:` `ML_PORT` (default `3003`).
- On each predict call the service logs at INFO: `predict: <n> task(s) [<t1>+<t2>+...] completed in <ms>ms` (e.g. `predict: 3 task(s) [clip+faces+ocr] completed in 142ms`). Always emitted (not gated by `ML_LOG_REQUESTS`).
- Service is launched via `start_service("ml", [py, "-m", "src.main"], env, ml_dir)` and logs to `~/.immich-accelerator/logs/ml.log`.
- `powermetrics` requires root; only it needs root, not the dashboard.
- Test fixtures available in `tests/conftest.py`: `tmp_data_dir`, `sample_config`, `saved_config`.

> ⚠️ Two regexes (the `powermetrics` field labels in Task 2 and the exact logger line prefix in Task 1) are written from the documented format but **must be re-verified against real output on the Mac Mini in Task 13.** The parsers are isolated pure functions specifically so this is a one-line adjustment if the real output differs.

---

## File Structure

- Create: `immich_accelerator/ml_stats.py` — pure parser for the ML request log (`parse_ml_log`).
- Create: `immich_accelerator/metrics.py` — `powermetrics` paths/content constants, `parse_powermetrics` (pure), `sample_powermetrics` (sudo invocation).
- Modify: `immich_accelerator/__main__.py` — `mode` branches in `cmd_setup`/`cmd_start`/`cmd_watch`/`cmd_status`/`cmd_logs`/`cmd_uninstall`, new helpers (`_setup_ml_only`, `_start_ml_only`, `_watch_ml_only`, `_ensure_dashboard_running`, `_detect_lan_ip`, `_print_nas_wiring`, `_install_powermetrics_sudoers`, `_remove_powermetrics_sudoers`), argparse flags.
- Modify: `immich_accelerator/dashboard.py` — `get_status` branch, `get_status_ml`, `_system_metrics`, `mode` field, requeue guard.
- Modify: `immich_accelerator/dashboard.html` — ml-only render path.
- Create: `tests/test_ml_stats.py`, `tests/test_metrics.py`.
- Modify: `tests/test_accelerator.py`, `tests/test_dashboard.py` — mode-branch + helper tests.
- Modify: `README.md` — appliance section, security note, Known differences row.

---

## Task 1: ML log parser (`ml_stats.parse_ml_log`)

**Files:**
- Create: `immich_accelerator/ml_stats.py`
- Test: `tests/test_ml_stats.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ml_stats.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ml_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'immich_accelerator.ml_stats'`.

- [ ] **Step 3: Write minimal implementation**

Create `immich_accelerator/ml_stats.py`:

```python
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
        if "face" in lowered:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ml_stats.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add immich_accelerator/ml_stats.py tests/test_ml_stats.py
git commit -m "feat: ML request-log parser for appliance dashboard

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: powermetrics parser + constants (`metrics.parse_powermetrics`)

**Files:**
- Create: `immich_accelerator/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
"""Tests for immich_accelerator.metrics — powermetrics parsing + sudoers content."""
from __future__ import annotations

from immich_accelerator import metrics

SAMPLE_PM = """\
*** Sampled system activity ***

**** GPU usage ****
GPU HW active residency:  37.50% (444 MHz: 12% 624 MHz: 25%)
GPU active residency:  37.50%
GPU Power: 1450 mW
ANE Power: 980 mW
Combined Power (CPU + GPU + ANE): 5200 mW
"""


class TestParsePowermetrics:
    def test_extracts_gpu_residency(self):
        assert metrics.parse_powermetrics(SAMPLE_PM)["gpu_residency_pct"] == 37.5

    def test_extracts_ane_power(self):
        assert metrics.parse_powermetrics(SAMPLE_PM)["ane_mw"] == 980.0

    def test_missing_fields_are_none(self):
        result = metrics.parse_powermetrics("nothing useful here")
        assert result == {"gpu_residency_pct": None, "ane_mw": None}


class TestSudoersContent:
    def test_sudoers_line_targets_wrapper(self):
        line = metrics.sudoers_content("alice")
        assert line.startswith("alice ALL=(root) NOPASSWD: ")
        assert str(metrics.POWERMETRICS_WRAPPER) in line
        assert line.endswith("\n")

    def test_wrapper_is_fixed_command_no_user_args(self):
        # The wrapper must hard-code the powermetrics invocation so the
        # sudoers grant cannot be abused with arbitrary args.
        assert "powermetrics" in metrics.WRAPPER_CONTENT
        assert metrics.WRAPPER_CONTENT.startswith("#!/bin/sh")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'immich_accelerator.metrics'`.

- [ ] **Step 3: Write minimal implementation**

Create `immich_accelerator/metrics.py`:

```python
"""Apple Silicon hardware metrics via powermetrics.

powermetrics requires root. We do NOT run the dashboard as root; instead
a fixed, root-owned wrapper script is installed and granted a scoped
passwordless sudoers rule (see __main__._install_powermetrics_sudoers).
The wrapper hard-codes the invocation so the grant can't be abused with
arbitrary args.

``parse_powermetrics`` is a pure function over the wrapper's stdout so it
is unit-testable. ``sample_powermetrics`` runs the wrapper via ``sudo -n``
(non-interactive — fails instead of prompting if the rule is absent).

NOTE: the field labels below are from documented powermetrics output;
re-verify against the Mac Mini (Task 13). ANE exposes power (mW), not a
utilization percentage.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

POWERMETRICS_WRAPPER = Path("/usr/local/sbin/immich-accelerator-powermetrics")
POWERMETRICS_SUDOERS = Path("/etc/sudoers.d/immich-accelerator")

WRAPPER_CONTENT = (
    "#!/bin/sh\n"
    "exec /usr/bin/powermetrics -n 1 -i 1000 --samplers gpu_power\n"
)

_GPU_RE = re.compile(r"GPU (?:HW )?active residency:\s+([\d.]+)%")
_ANE_RE = re.compile(r"ANE Power:\s+([\d.]+)\s*mW")


def sudoers_content(user: str) -> str:
    return f"{user} ALL=(root) NOPASSWD: {POWERMETRICS_WRAPPER}\n"


def parse_powermetrics(text: str) -> dict:
    gpu = _GPU_RE.search(text)
    ane = _ANE_RE.search(text)
    return {
        "gpu_residency_pct": float(gpu.group(1)) if gpu else None,
        "ane_mw": float(ane.group(1)) if ane else None,
    }


def sample_powermetrics() -> dict | None:
    """Run the privileged wrapper non-interactively; None if unavailable."""
    try:
        r = subprocess.run(
            ["sudo", "-n", str(POWERMETRICS_WRAPPER)],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if r.returncode != 0:
        return None
    return parse_powermetrics(r.stdout)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add immich_accelerator/metrics.py tests/test_metrics.py
git commit -m "feat: powermetrics parser and sudoers/wrapper content

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `sample_powermetrics` uses `sudo -n` (no prompt)

**Files:**
- Test: `tests/test_metrics.py` (add class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_metrics.py`:

```python
from unittest.mock import MagicMock, patch


class TestSamplePowermetrics:
    def test_invokes_wrapper_via_sudo_n(self):
        proc = MagicMock(returncode=0, stdout=SAMPLE_PM)
        with patch("immich_accelerator.metrics.subprocess.run", return_value=proc) as run:
            result = metrics.sample_powermetrics()
        cmd = run.call_args[0][0]
        assert cmd[:2] == ["sudo", "-n"]
        assert cmd[2] == str(metrics.POWERMETRICS_WRAPPER)
        assert result["gpu_residency_pct"] == 37.5

    def test_nonzero_return_yields_none(self):
        proc = MagicMock(returncode=1, stdout="")
        with patch("immich_accelerator.metrics.subprocess.run", return_value=proc):
            assert metrics.sample_powermetrics() is None

    def test_oserror_yields_none(self):
        with patch("immich_accelerator.metrics.subprocess.run", side_effect=OSError):
            assert metrics.sample_powermetrics() is None
```

- [ ] **Step 2: Run test to verify it passes (already implemented in Task 2)**

Run: `python -m pytest tests/test_metrics.py::TestSamplePowermetrics -v`
Expected: PASS (3 passed). If any fail, fix `sample_powermetrics` in `metrics.py` to match.

- [ ] **Step 3: Commit**

```bash
git add tests/test_metrics.py
git commit -m "test: cover sample_powermetrics sudo invocation paths

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: powermetrics sudoers/wrapper installer + remover

**Files:**
- Modify: `immich_accelerator/__main__.py` (add helpers; add imports `getpass`, `tempfile` if missing)
- Test: `tests/test_accelerator.py` (add class)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_accelerator.py` (import the new names in the existing import block):

```python
class TestPowermetricsInstaller:
    def test_validates_sudoers_before_install(self, tmp_path):
        import immich_accelerator.metrics as metrics
        from immich_accelerator.__main__ import _install_powermetrics_sudoers

        calls = []

        def fake_run(cmd, *a, **k):
            calls.append(cmd)
            return MagicMock(returncode=0, stderr="", stdout="")

        with patch("immich_accelerator.__main__.subprocess.run", side_effect=fake_run), \
             patch("immich_accelerator.__main__.getpass.getuser", return_value="bob"):
            ok = _install_powermetrics_sudoers()

        assert ok is True
        # visudo -cf must run before any `install` of the sudoers file
        joined = [" ".join(map(str, c)) for c in calls]
        visudo_idx = next(i for i, c in enumerate(joined) if "visudo -cf" in c)
        install_sudoers_idx = next(
            i for i, c in enumerate(joined)
            if "install" in c and str(metrics.POWERMETRICS_SUDOERS) in c
        )
        assert visudo_idx < install_sudoers_idx

    def test_aborts_when_visudo_fails(self):
        from immich_accelerator.__main__ import _install_powermetrics_sudoers

        def fake_run(cmd, *a, **k):
            if "visudo" in cmd:
                return MagicMock(returncode=1, stderr="bad", stdout="")
            return MagicMock(returncode=0, stderr="", stdout="")

        with patch("immich_accelerator.__main__.subprocess.run", side_effect=fake_run), \
             patch("immich_accelerator.__main__.getpass.getuser", return_value="bob"):
            assert _install_powermetrics_sudoers() is False

    def test_remove_deletes_both_paths(self):
        import immich_accelerator.metrics as metrics
        from immich_accelerator.__main__ import _remove_powermetrics_sudoers

        removed = []
        with patch("immich_accelerator.__main__.subprocess.run",
                   side_effect=lambda cmd, *a, **k: removed.append(cmd) or MagicMock(returncode=0)):
            _remove_powermetrics_sudoers()
        targets = {c[-1] for c in removed}
        assert str(metrics.POWERMETRICS_WRAPPER) in targets
        assert str(metrics.POWERMETRICS_SUDOERS) in targets
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_accelerator.py::TestPowermetricsInstaller -v`
Expected: FAIL — `ImportError: cannot import name '_install_powermetrics_sudoers'`.

- [ ] **Step 3: Write minimal implementation**

At the top of `immich_accelerator/__main__.py`, ensure these stdlib imports exist (add any missing): `getpass`, `tempfile`. Add `from . import metrics` near the other intra-package usage (or `import immich_accelerator.metrics as metrics`). Then add, near `_find_ml_dir` (around line 2789):

```python
def _install_powermetrics_sudoers() -> bool:
    """Install the root-owned powermetrics wrapper + a scoped NOPASSWD
    sudoers rule. Returns True on success. Requires interactive sudo
    (the user is prompted once)."""
    user = getpass.getuser()
    wrapper = metrics.POWERMETRICS_WRAPPER
    sudoers = metrics.POWERMETRICS_SUDOERS

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as wf:
        wf.write(metrics.WRAPPER_CONTENT)
        wtmp = wf.name
    with tempfile.NamedTemporaryFile("w", suffix=".sudoers", delete=False) as sf:
        sf.write(metrics.sudoers_content(user))
        stmp = sf.name

    log.info("Configuring powermetrics access (sudo required, one time)...")
    try:
        chk = subprocess.run(
            ["sudo", "visudo", "-cf", stmp], capture_output=True, text=True
        )
        if chk.returncode != 0:
            log.error("sudoers validation failed: %s", chk.stderr.strip())
            return False
        subprocess.run(
            ["sudo", "install", "-d", "-m", "755", "-o", "root", "-g", "wheel",
             str(wrapper.parent)],
            check=True,
        )
        subprocess.run(
            ["sudo", "install", "-m", "755", "-o", "root", "-g", "wheel",
             wtmp, str(wrapper)],
            check=True,
        )
        subprocess.run(
            ["sudo", "install", "-m", "440", "-o", "root", "-g", "wheel",
             stmp, str(sudoers)],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        log.error("powermetrics setup failed: %s", e)
        return False
    finally:
        for p in (wtmp, stmp):
            try:
                os.unlink(p)
            except OSError:
                pass


def _remove_powermetrics_sudoers() -> None:
    """Remove the powermetrics wrapper + sudoers rule (best effort)."""
    for p in (metrics.POWERMETRICS_SUDOERS, metrics.POWERMETRICS_WRAPPER):
        subprocess.run(["sudo", "rm", "-f", str(p)], capture_output=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_accelerator.py::TestPowermetricsInstaller -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add immich_accelerator/__main__.py tests/test_accelerator.py
git commit -m "feat: install/remove scoped powermetrics sudoers rule

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: LAN IP detection + NAS wiring print

**Files:**
- Modify: `immich_accelerator/__main__.py` (add `_detect_lan_ip`, `_print_nas_wiring`)
- Test: `tests/test_accelerator.py` (add class)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_accelerator.py`:

```python
class TestLanIp:
    def test_returns_first_iface_with_address(self):
        from immich_accelerator.__main__ import _detect_lan_ip

        def fake_run(cmd, *a, **k):
            iface = cmd[-1]
            out = "192.168.1.42\n" if iface == "en0" else ""
            return MagicMock(returncode=0, stdout=out)

        with patch("immich_accelerator.__main__.subprocess.run", side_effect=fake_run):
            assert _detect_lan_ip() == "192.168.1.42"

    def test_returns_none_when_no_address(self):
        from immich_accelerator.__main__ import _detect_lan_ip
        with patch("immich_accelerator.__main__.subprocess.run",
                   return_value=MagicMock(returncode=1, stdout="")):
            assert _detect_lan_ip() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_accelerator.py::TestLanIp -v`
Expected: FAIL — `ImportError: cannot import name '_detect_lan_ip'`.

- [ ] **Step 3: Write minimal implementation**

Add to `immich_accelerator/__main__.py` near the other helpers:

```python
def _detect_lan_ip() -> str | None:
    """Best-effort LAN IPv4 for this Mac (en0 then en1)."""
    for iface in ("en0", "en1"):
        try:
            r = subprocess.run(
                ["ipconfig", "getifaddr", iface],
                capture_output=True, text=True, timeout=3,
            )
            ip = r.stdout.strip()
            if r.returncode == 0 and ip:
                return ip
        except (subprocess.SubprocessError, OSError):
            pass
    return None


def _print_nas_wiring(port: int) -> None:
    """Print the exact change to make on the NAS / remote Immich host."""
    ip = _detect_lan_ip() or "<this-mac-LAN-ip>"
    log.info("")
    log.info("On your NAS (remote Immich), point ML at this Mac:")
    log.info("  IMMICH_MACHINE_LEARNING_URL=http://%s:%d", ip, port)
    log.info("Then stop the old Docker ML container:")
    log.info("  docker stop immich-machine-learning  (or remove it from compose)")
    log.info("")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_accelerator.py::TestLanIp -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add immich_accelerator/__main__.py tests/test_accelerator.py
git commit -m "feat: detect Mac LAN IP and print NAS wiring instructions

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `setup --ml-only` (flag, dispatch, `_setup_ml_only`)

**Files:**
- Modify: `immich_accelerator/__main__.py` (argparse, `cmd_setup` dispatch, `_setup_ml_only`)
- Test: `tests/test_accelerator.py` (add class)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_accelerator.py`:

```python
class TestSetupMlOnly:
    def test_writes_minimal_ml_only_config(self, tmp_data_dir):
        from immich_accelerator.__main__ import _setup_ml_only, load_config

        args = argparse.Namespace(ml_only=True, port=3003, host="0.0.0.0",
                                  url=None, api_key=None, manual=False,
                                  import_server=None)
        with patch("immich_accelerator.__main__._find_ml_dir",
                   return_value=Path("/Users/test/ml")), \
             patch("immich_accelerator.__main__._install_powermetrics_sudoers",
                   return_value=True), \
             patch("immich_accelerator.__main__._print_nas_wiring"):
            _setup_ml_only(args)

        cfg = load_config()
        assert cfg["mode"] == "ml-only"
        assert cfg["ml_dir"] == "/Users/test/ml"
        assert cfg["ml_host"] == "0.0.0.0"
        assert cfg["ml_port"] == 3003
        assert cfg["metrics_powermetrics"] is True
        # appliance config must NOT carry worker/db keys
        for k in ("db_password", "redis_port", "server_dir", "upload_mount"):
            assert k not in cfg

    def test_disables_metrics_when_sudoers_fails(self, tmp_data_dir):
        from immich_accelerator.__main__ import _setup_ml_only, load_config

        args = argparse.Namespace(ml_only=True, port=3003, host="0.0.0.0",
                                  url=None, api_key=None, manual=False,
                                  import_server=None)
        with patch("immich_accelerator.__main__._find_ml_dir",
                   return_value=Path("/Users/test/ml")), \
             patch("immich_accelerator.__main__._install_powermetrics_sudoers",
                   return_value=False), \
             patch("immich_accelerator.__main__._print_nas_wiring"):
            _setup_ml_only(args)

        assert load_config()["metrics_powermetrics"] is False

    def test_cmd_setup_dispatches_to_ml_only(self, tmp_data_dir):
        from immich_accelerator.__main__ import cmd_setup

        args = argparse.Namespace(ml_only=True, port=3003, host="0.0.0.0",
                                  url=None, api_key=None, manual=False,
                                  import_server=None)
        with patch("immich_accelerator.__main__._setup_ml_only") as m:
            cmd_setup(args)
        m.assert_called_once_with(args)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_accelerator.py::TestSetupMlOnly -v`
Expected: FAIL — `ImportError: cannot import name '_setup_ml_only'`.

- [ ] **Step 3: Write minimal implementation**

In `main()` argparse, modify the `setup` parser (around line 3790) to add flags:

```python
    setup_p.add_argument("--ml-only", dest="ml_only", action="store_true",
                         help="Set up ML appliance mode (Metal ML endpoint only)")
    setup_p.add_argument("--port", type=int, default=3003,
                         help="ML service port (ml-only mode)")
    setup_p.add_argument("--host", default="0.0.0.0",
                         help="ML service bind host (ml-only mode)")
```

At the top of `cmd_setup` (line 2740, before the `if args.manual:` branch):

```python
    if getattr(args, "ml_only", False):
        return _setup_ml_only(args)
```

Add the helper near `_find_ml_dir`:

```python
def _setup_ml_only(args) -> None:
    """Configure ML appliance mode: native Metal ML service as a remote
    ML endpoint. No Docker / DB / worker / shared filesystem."""
    ml_dir = _find_ml_dir()
    if not ml_dir:
        raise RuntimeError(
            "ML service unavailable — cannot set up ml-only mode. "
            "Ensure Python 3.11+ and the ml/ submodule are present."
        )
    config = {
        "mode": "ml-only",
        "ml_dir": str(ml_dir),
        "ml_host": getattr(args, "host", None) or "0.0.0.0",
        "ml_port": int(getattr(args, "port", None) or 3003),
        "metrics_powermetrics": True,
        "dashboard_port": 8420,
    }
    save_config(config)
    log.info("Wrote ml-only config to %s", CONFIG_FILE)

    if _install_powermetrics_sudoers():
        log.info("Real GPU/ANE metrics enabled (powermetrics).")
    else:
        config["metrics_powermetrics"] = False
        save_config(config)
        log.warning("Continuing without real GPU/ANE metrics.")

    _print_nas_wiring(config["ml_port"])
    log.info("Setup complete. Start with: immich-accelerator start")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_accelerator.py::TestSetupMlOnly -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add immich_accelerator/__main__.py tests/test_accelerator.py
git commit -m "feat: add 'setup --ml-only' appliance configuration

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `start` branch (`_start_ml_only`) + dashboard helper

**Files:**
- Modify: `immich_accelerator/__main__.py` (`cmd_start` branch, `_start_ml_only`, `_ensure_dashboard_running`)
- Test: `tests/test_accelerator.py` (add class)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_accelerator.py`:

```python
class TestStartMlOnly:
    def _make_ml_dir(self, tmp_path):
        ml = tmp_path / "ml"
        (ml / "venv" / "bin").mkdir(parents=True)
        (ml / "venv" / "bin" / "python3").write_text("#!/bin/sh\n")
        return ml

    def test_cmd_start_dispatches_to_ml_only(self, tmp_data_dir, tmp_path):
        from immich_accelerator.__main__ import cmd_start, save_config
        save_config({"mode": "ml-only", "ml_dir": str(tmp_path / "ml"),
                     "ml_host": "0.0.0.0", "ml_port": 3003})
        with patch("immich_accelerator.__main__._start_ml_only") as m:
            cmd_start(argparse.Namespace(force=False))
        assert m.call_count == 1

    def test_start_ml_only_launches_with_ml_env(self, tmp_data_dir, tmp_path):
        from immich_accelerator.__main__ import _start_ml_only
        ml = self._make_ml_dir(tmp_path)
        config = {"mode": "ml-only", "ml_dir": str(ml),
                  "ml_host": "0.0.0.0", "ml_port": 3055}
        with patch("immich_accelerator.__main__._kill_stale_processes"), \
             patch("immich_accelerator.__main__.read_pid", return_value=None), \
             patch("immich_accelerator.__main__._ensure_dashboard_running"), \
             patch("immich_accelerator.__main__.start_service", return_value=4242) as ss:
            _start_ml_only(config, argparse.Namespace(force=False))
        name, cmd, env, cwd = ss.call_args[0]
        assert name == "ml"
        assert cmd == [str(ml / "venv" / "bin" / "python3"), "-m", "src.main"]
        assert env["ML_HOST"] == "0.0.0.0"
        assert env["ML_PORT"] == "3055"
        assert cwd == str(ml)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_accelerator.py::TestStartMlOnly -v`
Expected: FAIL — `ImportError: cannot import name '_start_ml_only'`.

- [ ] **Step 3: Write minimal implementation**

At the top of `cmd_start` (line 2954, right after `config = load_config()`):

```python
    if config.get("mode") == "ml-only":
        return _start_ml_only(config, args)
```

Add helpers (near `cmd_start`):

```python
def _ensure_dashboard_running(config: dict) -> None:
    """Start the dashboard in the background if it isn't already up."""
    import urllib.request as _urlreq
    port = int(config.get("dashboard_port", 8420))
    try:
        _urlreq.urlopen(f"http://localhost:{port}/", timeout=2)
        return  # already running
    except Exception:
        pass
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    dash_log = open(LOG_DIR / "dashboard.log", "a")
    proc = subprocess.Popen(
        [sys.executable, "-m", __package__ or "immich_accelerator",
         "dashboard", "--port", str(port)],
        cwd=str(Path(__file__).parent.parent),
        stdout=dash_log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    dash_log.close()
    write_pid("dashboard", proc.pid)
    log.info("Dashboard started: http://localhost:%d", port)


def _start_ml_only(config: dict, args) -> None:
    """Start only the native Metal ML service (+ dashboard)."""
    _kill_stale_processes()

    if read_pid("ml") and not getattr(args, "force", False):
        log.info("ML service already running")
    else:
        if getattr(args, "force", False):
            kill_pid("ml")
        ml_dir = Path(config.get("ml_dir", ""))
        if not (ml_dir / "venv" / "bin" / "python3").exists():
            resolved = _find_ml_dir()
            if resolved:
                ml_dir = resolved
                config["ml_dir"] = str(resolved)
                save_config(config)
        ml_python = ml_dir / "venv" / "bin" / "python3"
        if not ml_python.exists():
            raise RuntimeError(
                "ML venv not found — run: immich-accelerator setup --ml-only"
            )
        env = os.environ.copy()
        env["ML_HOST"] = config.get("ml_host", "0.0.0.0")
        env["ML_PORT"] = str(config.get("ml_port", 3003))
        pid = start_service("ml", [str(ml_python), "-m", "src.main"],
                            env, str(ml_dir))
        log.info("ML service running (PID %d) on %s:%s",
                 pid, env["ML_HOST"], env["ML_PORT"])

    _ensure_dashboard_running(config)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_accelerator.py::TestStartMlOnly -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add immich_accelerator/__main__.py tests/test_accelerator.py
git commit -m "feat: ml-only start path (ML service + dashboard, no worker)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `watch` / `status` / `logs` ml-only branches

**Files:**
- Modify: `immich_accelerator/__main__.py` (`cmd_watch`, `cmd_status`, `cmd_logs`, `_watch_ml_only`)
- Test: `tests/test_accelerator.py` (add class)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_accelerator.py`:

```python
class TestMlOnlyCommands:
    def test_watch_dispatches_to_ml_only(self, tmp_data_dir):
        from immich_accelerator.__main__ import cmd_watch, save_config
        save_config({"mode": "ml-only", "ml_dir": "/x", "ml_port": 3003})
        with patch("immich_accelerator.__main__._watch_ml_only") as m:
            cmd_watch(None)
        m.assert_called_once()

    def test_status_ml_only_reports_endpoint(self, tmp_data_dir, capsys, caplog):
        import logging
        from immich_accelerator.__main__ import cmd_status, save_config
        save_config({"mode": "ml-only", "ml_host": "0.0.0.0", "ml_port": 3003})
        with patch("immich_accelerator.__main__.read_pid", return_value=1234), \
             caplog.at_level(logging.INFO):
            cmd_status(None)
        text = caplog.text
        assert "ml-only" in text
        assert "3003" in text

    def test_logs_defaults_to_ml_in_ml_only(self, tmp_data_dir):
        from immich_accelerator.__main__ import cmd_logs, save_config, LOG_DIR
        save_config({"mode": "ml-only", "ml_port": 3003})
        # no ml.log present -> prints "No log file" and returns (no exec)
        with patch("immich_accelerator.__main__.os.execvp") as ex:
            cmd_logs(argparse.Namespace(service=None))
        ex.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_accelerator.py::TestMlOnlyCommands -v`
Expected: FAIL — `ImportError: cannot import name '_watch_ml_only'` (and/or assertion failures).

- [ ] **Step 3: Write minimal implementation**

`cmd_watch` — at the very top (after the docstring, before the existing body around line 3344):

```python
    if load_config().get("mode") == "ml-only":
        return _watch_ml_only()
```

Add `_watch_ml_only` near `cmd_watch`:

```python
def _watch_ml_only() -> None:
    """KeepAlive monitor for appliance mode: ML service + dashboard only."""
    log.info("Watching ML appliance (Ctrl+C to stop)...")
    config = load_config()
    if not read_pid("ml"):
        log.info("ML not running, starting...")
        cmd_start(argparse.Namespace(force=True))
    _ensure_dashboard_running(config)
    while True:
        try:
            time.sleep(30)
            config = load_config()
            if not read_pid("ml"):
                log.warning("ML service not running — restarting...")
                try:
                    _start_ml_only(config, argparse.Namespace(force=True))
                except RuntimeError:
                    log.error("  ML restart failed, will retry in 30s")
            _ensure_dashboard_running(config)
        except KeyboardInterrupt:
            log.info("Watch stopped")
            return
```

`cmd_status` — restructure to load config first and branch. Replace the body (lines 3271-3287) with:

```python
    config = load_config() if CONFIG_FILE.exists() else {}

    if config.get("mode") == "ml-only":
        ml_pid = read_pid("ml")
        log.info("Mode:       ml-only (ML appliance)")
        log.info("ML service: %s",
                 f"running (PID {ml_pid})" if ml_pid else "stopped")
        log.info("Endpoint:   http://%s:%s",
                 config.get("ml_host", "0.0.0.0"), config.get("ml_port", 3003))
        return

    worker_pid = read_pid("worker")
    ml_pid = read_pid("ml")
    if not worker_pid and not ml_pid:
        log.info("Not running")
        return
    log.info("Worker:     %s",
             f"running (PID {worker_pid})" if worker_pid else "stopped")
    log.info("ML service: %s", f"running (PID {ml_pid})" if ml_pid else "stopped")
    if config:
        log.info("Version:    %s", config.get("version", "?"))
        if config.get("ffmpeg_path"):
            log.info("FFmpeg:     %s (VideoToolbox)", config["ffmpeg_path"])
```

`cmd_logs` — change the default target (line 3291):

```python
def cmd_logs(args):
    default = "ml" if (CONFIG_FILE.exists() and load_config().get("mode") == "ml-only") else "worker"
    target = args.service or default
    log_file = LOG_DIR / f"{target}.log"
    if not log_file.exists():
        print(f"No log file: {log_file}")
        return
    os.execvp("tail", ["tail", "-f", str(log_file)])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_accelerator.py::TestMlOnlyCommands -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add immich_accelerator/__main__.py tests/test_accelerator.py
git commit -m "feat: ml-only watch/status/logs branches

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: `uninstall` removes powermetrics artifacts

**Files:**
- Modify: `immich_accelerator/__main__.py` (`cmd_uninstall`)
- Test: `tests/test_accelerator.py` (add test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_accelerator.py`:

```python
class TestUninstallPowermetrics:
    def test_uninstall_removes_powermetrics(self, tmp_data_dir, monkeypatch):
        from immich_accelerator.__main__ import cmd_uninstall
        monkeypatch.setattr("builtins.input", lambda *_: "y")
        with patch("immich_accelerator.__main__.cmd_stop"), \
             patch("immich_accelerator.__main__._remove_build_link"), \
             patch("immich_accelerator.__main__._rmtree_or_explain", return_value=True), \
             patch("immich_accelerator.__main__.subprocess.run", return_value=MagicMock(returncode=0, stdout="")), \
             patch("immich_accelerator.__main__._remove_powermetrics_sudoers") as rm:
            cmd_uninstall(None)
        rm.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_accelerator.py::TestUninstallPowermetrics -v`
Expected: FAIL — `_remove_powermetrics_sudoers` not called.

- [ ] **Step 3: Write minimal implementation**

In `cmd_uninstall`, after the launchd plist removal and before/after `_remove_build_link()` (around line 3750), add:

```python
    # Remove powermetrics wrapper + sudoers rule (ml-only mode)
    _remove_powermetrics_sudoers()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_accelerator.py::TestUninstallPowermetrics -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add immich_accelerator/__main__.py tests/test_accelerator.py
git commit -m "feat: uninstall removes powermetrics wrapper and sudoers rule

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Dashboard backend — `get_status_ml`, `_system_metrics`, mode field, requeue guard

**Files:**
- Modify: `immich_accelerator/dashboard.py`
- Test: `tests/test_dashboard.py` (add class)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_dashboard.py`:

```python
class TestGetStatusMl:
    def _cfg(self):
        return {"mode": "ml-only", "ml_host": "0.0.0.0", "ml_port": 3003,
                "metrics_powermetrics": True}

    def test_routes_to_ml_status(self):
        import immich_accelerator.dashboard as dash
        with patch.object(dash, "get_status_ml", return_value={"mode": "ml-only"}) as m:
            out = dash.get_status(self._cfg())
        assert out == {"mode": "ml-only"}
        m.assert_called_once()

    def test_ml_status_shape(self):
        import immich_accelerator.dashboard as dash
        sample_log = "predict: 1 task(s) [clip] completed in 40ms\n"
        with patch.object(dash, "_tail_text", return_value=sample_log), \
             patch.object(dash, "_ping_ml", return_value=True), \
             patch("immich_accelerator.metrics.sample_powermetrics",
                   return_value={"gpu_residency_pct": 30.0, "ane_mw": 500.0}), \
             patch.object(dash, "_system_metrics",
                          return_value={"load_1m": 1.0, "mem_total_gb": 24.0, "cpus": 10}):
            # bypass the ml-cache so the body runs
            dash._ml_cache = None
            dash._ml_cache_ts = 0
            out = dash.get_status_ml(self._cfg())
        assert out["mode"] == "ml-only"
        assert out["services"]["ml"]["alive"] is True
        assert out["ml"]["tasks"] == {"clip": 1, "faces": 0, "ocr": 0}
        assert out["hardware"]["gpu_residency_pct"] == 30.0
        assert out["hardware"]["ane_mw"] == 500.0
        assert out["hardware"]["powermetrics"] is True

    def test_full_status_carries_mode_field(self, ):
        import immich_accelerator.dashboard as dash
        dash._cache = None
        dash._cache_ts = 0
        with patch.object(dash, "_query_db", return_value="0|0|0|0|0|0|0"), \
             patch.object(dash, "_run", return_value=""), \
             patch("urllib.request.urlopen", side_effect=Exception):
            out = dash.get_status({"mode": "full"})
        assert out["mode"] == "full"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dashboard.py::TestGetStatusMl -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'get_status_ml'`.

- [ ] **Step 3: Write minimal implementation**

In `immich_accelerator/dashboard.py`:

Add module globals near the existing `_cache`/`_cache_ts` declarations:

```python
_ml_cache = None
_ml_cache_ts = 0.0
_ml_last_total = 0
_ml_last_ts = 0.0
```

Add the early branch at the very top of `get_status` (line 142, before the existing cache check):

```python
    if config.get("mode") == "ml-only":
        return get_status_ml(config)
```

Add `"mode": "full",` as a key in the existing full-status `status = {...}` dict (around line 305).

Add these helpers (after `get_status`):

```python
def _ping_ml(config: dict) -> bool:
    import urllib.request as _urlreq
    port = int(config.get("ml_port", 3003))
    try:
        with _urlreq.urlopen(f"http://localhost:{port}/ping", timeout=2) as r:
            return r.read().decode().strip() == "pong"
    except Exception:
        return False


def _tail_text(path: Path, max_bytes: int = 65536) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _system_metrics() -> dict:
    load_raw = _run(["sysctl", "-n", "vm.loadavg"])
    load_1m = 0.0
    if load_raw:
        try:
            load_1m = float(load_raw.strip("{ }").split()[0])
        except (ValueError, IndexError):
            pass
    mem_raw = _run(["sysctl", "-n", "hw.memsize"])
    cpu_raw = _run(["sysctl", "-n", "hw.ncpu"])
    return {
        "load_1m": load_1m,
        "mem_total_gb": round(int(mem_raw) / (1024**3), 1) if mem_raw else 0,
        "cpus": int(cpu_raw) if cpu_raw else 0,
    }


def get_status_ml(config: dict) -> dict:
    """ML appliance status: health, throughput, latency, real GPU/ANE."""
    global _ml_cache, _ml_cache_ts, _ml_last_total, _ml_last_ts
    from . import metrics
    from .ml_stats import parse_ml_log

    now = time.monotonic()
    if _ml_cache and now - _ml_cache_ts < _CACHE_TTL:
        return _ml_cache

    ml_alive = _ping_ml(config)
    log_path = Path.home() / ".immich-accelerator" / "logs" / "ml.log"
    stats = parse_ml_log(_tail_text(log_path))

    rate = 0.0
    if _ml_last_ts and now > _ml_last_ts:
        rate = max(0.0, (stats["total"] - _ml_last_total) / (now - _ml_last_ts))
    _ml_last_total = stats["total"]
    _ml_last_ts = now

    pm = metrics.sample_powermetrics() if config.get("metrics_powermetrics") else None

    status = {
        "mode": "ml-only",
        "services": {"ml": {"alive": ml_alive, "name": "ML Service"}},
        "ml": {
            "throughput_rps": round(rate, 2),
            "tasks": stats["tasks"],
            "latency_ms": stats["latency_ms"],
            "total_predicts": stats["total"],
            "endpoint": f"http://{config.get('ml_host', '0.0.0.0')}:{config.get('ml_port', 3003)}",
        },
        "hardware": {
            "gpu_residency_pct": pm.get("gpu_residency_pct") if pm else None,
            "ane_mw": pm.get("ane_mw") if pm else None,
            "powermetrics": bool(pm),
        },
        "system": _system_metrics(),
        "version": "—",
        "accelerator_version": _get_accelerator_version(),
    }
    _ml_cache = status
    _ml_cache_ts = now
    return status
```

In `create_app`, guard `api_requeue` (top of the function body, line 367):

```python
        if config.get("mode") == "ml-only":
            return JSONResponse(
                {"error": "requeue disabled in ml-only mode"}, status_code=400
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dashboard.py::TestGetStatusMl -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python -m pytest -q`
Expected: all pass (DB-marked tests may skip).

- [ ] **Step 6: Commit**

```bash
git add immich_accelerator/dashboard.py tests/test_dashboard.py
git commit -m "feat: ML-appliance dashboard status (throughput + real GPU/ANE)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Dashboard frontend — ml-only layout

**Files:**
- Modify: `immich_accelerator/dashboard.html`

This task is visual; verify in a browser (Task 13 covers the Mac Mini run; here verify rendering locally against a stubbed `/api/status`).

- [ ] **Step 1: Read the file and locate the render entry point**

Run: `python -m pytest -q` (sanity) then open `immich_accelerator/dashboard.html`. Identify:
- the `fetch('/api/status')` handler / render function (search `api/status`),
- the existing "Metal GPU" and "Neural Engine" chip cards (search `chip-header gpu` and `chip-header ane`, ~lines 296-297),
- the section containers for worker/queue/video/progress panels (search `progress`, `queue`, `Microservices`).

- [ ] **Step 2: Add an ml-only render path**

In the status-render JS, branch on `data.mode`. Add a function and call it when `data.mode === 'ml-only'`:

```javascript
function renderMl(data) {
  // Hide full-mode panels (worker/queue/video/progress). Give those
  // top-level section containers id="full-only" or add the class
  // "full-only" to each during Step 1, then:
  document.querySelectorAll('.full-only').forEach(el => el.style.display = 'none');

  const ml = data.ml || {};
  const hw = data.hardware || {};
  const lat = ml.latency_ms || {};

  // Service health
  setText('ml-health', data.services?.ml?.alive ? 'online' : 'offline');
  setText('ml-endpoint', ml.endpoint || '');
  setText('ml-throughput', (ml.throughput_rps ?? 0).toFixed(2) + ' req/s');
  setText('ml-latency', 'p50 ' + (lat.p50 ?? 0) + 'ms · max ' + (lat.max ?? 0) + 'ms');

  // GPU panel (real residency %)
  const gpuPct = hw.gpu_residency_pct;
  setBar('gpu', gpuPct == null ? 0 : gpuPct);
  setText('gpu-stat', gpuPct == null
    ? 'powermetrics unavailable'
    : gpuPct.toFixed(1) + '% · CLIP ' + (ml.tasks?.clip ?? 0));

  // ANE panel (power mW, no % — show active + mW)
  const aneMw = hw.ane_mw;
  setBar('ane', aneMw == null ? 0 : Math.min(100, aneMw / 80)); // visual only
  setText('ane-stat', aneMw == null
    ? 'powermetrics unavailable'
    : aneMw.toFixed(0) + ' mW · faces ' + (ml.tasks?.faces ?? 0) + ' · ocr ' + (ml.tasks?.ocr ?? 0));
}

// Small helpers (reuse existing ones if present):
function setText(id, v){ const e=document.getElementById(id); if(e) e.textContent=v; }
function setBar(cls, pct){ const e=document.querySelector('.bar-fill.'+cls); if(e) e.style.width=Math.max(0,Math.min(100,pct))+'%'; }
```

Wire it where the fetched data is handled:

```javascript
  if (data.mode === 'ml-only') { renderMl(data); return; }
  // ...existing full-mode rendering below...
```

In the markup, add the `full-only` class to the worker/queue/video/progress section containers, and add the ids referenced above (`ml-health`, `ml-endpoint`, `ml-throughput`, `ml-latency`, `gpu-stat`, `ane-stat`) to the relevant elements. Reuse the existing `chip-header gpu` / `chip-header ane` cards for the GPU/ANE panels.

- [ ] **Step 3: Verify rendering locally**

Run the dashboard against a temporary ml-only config:

```bash
python - <<'PY'
import json, os, pathlib
d = pathlib.Path.home()/".immich-accelerator"; d.mkdir(exist_ok=True)
(d/"config.json").write_text(json.dumps({"mode":"ml-only","ml_host":"0.0.0.0","ml_port":3003,"metrics_powermetrics":False}))
print("wrote ml-only config")
PY
python -m immich_accelerator dashboard --port 8420
```

Open `http://localhost:8420`. Expected: GPU/ANE/throughput panels render; worker/queue panels are hidden; no JS console errors. (GPU/ANE show "powermetrics unavailable" since `metrics_powermetrics:false` here — that's correct.) Stop with Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add immich_accelerator/dashboard.html
git commit -m "feat: ML-appliance dashboard layout (GPU/ANE/throughput panels)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the appliance section**

After the "Split deployment" section (around line 183), add:

```markdown
## ML appliance mode (remote ML endpoint for a NAS)

If your Immich server runs elsewhere (e.g. a NAS) and you only want this
Mac to provide **GPU-accelerated machine learning**, run the accelerator in
ML appliance mode. The Mac runs only the native Metal/ANE ML service as a
drop-in replacement for Immich's Docker ML container — no worker, no shared
filesystem, no database access on the Mac.

```bash
immich-accelerator setup --ml-only        # optional: --port 3003 --host 0.0.0.0
immich-accelerator start
```

Setup prints the exact line to set on your NAS:

```
IMMICH_MACHINE_LEARNING_URL=http://<mac-LAN-ip>:3003
```

Then stop your old Docker `immich-machine-learning` container. ML inference
is pure HTTP (images in, embeddings out) — no shared storage is required.

The dashboard (`http://<mac>:8420`) shows ML health, request throughput and
latency, and real Metal GPU residency / ANE power.

### Security note

The ML service has **no authentication** (Immich's ML never has) and binds
`0.0.0.0` by default, so any host on your LAN can call it. Keep it on a
trusted network. To bind a single interface, set `"ml_host"` to that IP in
`~/.immich-accelerator/config.json`. Real GPU/ANE metrics use `powermetrics`,
which requires root; setup installs a scoped passwordless `sudoers` rule
pointing at a fixed, root-owned wrapper (`uninstall` removes both).
```

- [ ] **Step 2: Add a Known differences row**

In the "Known differences" table, add:

```markdown
| **ML appliance mode** | ML + worker on one host | ML service only, worker stays on the remote Immich host | Pure HTTP ML endpoint; no shared filesystem needed. |
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document ML appliance mode and security posture

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Mac Mini end-to-end verification + regex finalization

**Files:** none (verification); small follow-up edits to `metrics.py`/`ml_stats.py` only if real output differs.

Per CLAUDE.md: deploy to the Mac Mini (`ssh macmini`) and verify before claiming anything works. Do NOT skip.

- [ ] **Step 1: Capture real output to confirm the two flagged regexes**

```bash
ssh macmini 'sudo /usr/local/sbin/immich-accelerator-powermetrics' | sed -n '1,60p'
ssh macmini 'tail -n 40 ~/.immich-accelerator/logs/ml.log'
```

Confirm the `powermetrics` lines match `_GPU_RE`/`_ANE_RE` in `metrics.py` (especially that **ANE Power appears under the `gpu_power` sampler**; if not, change `WRAPPER_CONTENT` to `--samplers cpu_power,gpu_power`). Confirm the predict line matches `_PREDICT_RE` in `ml_stats.py`. If either differs, update the regex/sampler, re-run the relevant unit test with a fixture captured from real output, and commit.

- [ ] **Step 2: Full appliance bring-up**

```bash
ssh macmini 'immich-accelerator setup --ml-only && immich-accelerator start && immich-accelerator status'
```

Expected: status shows `Mode: ml-only`, ML running, endpoint printed.

- [ ] **Step 3: LAN reachability + real predict**

From another host on the LAN:

```bash
curl -s http://<mac-LAN-ip>:3003/ping          # -> pong
immich-accelerator ml-test                       # on the mac: all checks pass
```

- [ ] **Step 4: Wire the NAS and confirm jobs process**

Set `IMMICH_MACHINE_LEARNING_URL=http://<mac-LAN-ip>:3003` on the NAS, stop the Docker ML container, restart the NAS Immich server. Then, using the Immich API (not assumptions, per CLAUDE.md), confirm Smart Search / Face Detection / OCR jobs complete:

```bash
curl -s -H "x-api-key: $KEY" http://<nas>:2283/api/jobs | python -m json.tool
```

- [ ] **Step 5: Dashboard under load**

Open `http://<mac>:8420` while jobs run. Confirm: ML online, throughput > 0, latency populated, **real** GPU residency moves during CLIP work, ANE shows mW during face/OCR work. Capture a Playwright screenshot (per CLAUDE.md).

- [ ] **Step 6: launchd resilience**

```bash
ssh macmini 'kill $(cat ~/.immich-accelerator/pids/ml.pid)'
# wait ~35s
ssh macmini 'immich-accelerator status'   # ML running again (watch restarted it)
```

- [ ] **Step 7: Commit any regex/sampler fixes from Step 1**

```bash
git add immich_accelerator/metrics.py immich_accelerator/ml_stats.py tests/
git commit -m "fix: align powermetrics/ml-log parsers with Mac Mini output

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run the full test suite: `python -m pytest -q` — all pass.
- [ ] `git log --oneline` shows one clean commit per task on `main`.
- [ ] `git push origin main`.
- [ ] Confirm the appliance still serves the NAS (Immich API job counts increasing).

---

## Self-review notes (spec coverage)

- Spec §Config → Task 6. §setup --ml-only → Tasks 5,6. §start/watch/status/logs → Tasks 7,8. §dashboard (backend) → Task 10; (frontend) → Task 11. §powermetrics+sudoers → Tasks 2,3,4; uninstall → Task 9. §security → Task 12. §throughput parser → Task 1. §NAS wiring → Task 5. §Testing(pytest) → Tasks 1-10; §Testing(E2E) → Task 13. §Docs → Task 12.
- `ml-test` unchanged (already mode-agnostic) — covered by Task 13 Step 3.
- `stop` needs no change (already iterates worker/ml/dashboard) — intentionally omitted.
