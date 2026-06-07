"""Tests for immich_accelerator.metrics — powermetrics plist parsing + sudoers."""

from __future__ import annotations

import plistlib
from unittest.mock import MagicMock, patch

from immich_accelerator import metrics


def _plist(gpu=None, processor=None) -> bytes:
    """Build a powermetrics-style plist sample (XML plist bytes)."""
    d = {}
    if gpu is not None:
        d["gpu"] = gpu
    if processor is not None:
        d["processor"] = processor
    return plistlib.dumps(d)


# A faithful reproduction of one real sample captured during Task 13 on
# Apple Silicon / macOS 26 (Mac17,9): `powermetrics --samplers
# cpu_power,gpu_power -f plist`. GPU residency derives from gpu.idle_ratio;
# ANE power is processor.ane_power.
REAL_SAMPLE = _plist(
    gpu={"freq_hz": 338.0, "idle_ns": 291485041, "idle_ratio": 0.897279, "gpu_energy": 36},
    processor={
        "cpu_power": 1182.32,
        "gpu_power": 112.901,
        "ane_power": 0.0,
        "combined_power": 1295.22,
    },
)


class TestParsePowermetrics:
    def test_gpu_residency_from_idle_ratio(self):
        # (1 - 0.897279) * 100 = 10.2721 -> 10.27
        assert metrics.parse_powermetrics(REAL_SAMPLE)["gpu_residency_pct"] == 10.27

    def test_ane_power(self):
        assert metrics.parse_powermetrics(REAL_SAMPLE)["ane_mw"] == 0.0

    def test_ane_power_nonzero(self):
        raw = _plist(processor={"ane_power": 980.0})
        assert metrics.parse_powermetrics(raw)["ane_mw"] == 980.0

    def test_fully_idle_gpu(self):
        raw = _plist(gpu={"idle_ratio": 1.0})
        assert metrics.parse_powermetrics(raw)["gpu_residency_pct"] == 0.0

    def test_fully_busy_gpu(self):
        raw = _plist(gpu={"idle_ratio": 0.0})
        assert metrics.parse_powermetrics(raw)["gpu_residency_pct"] == 100.0

    def test_handles_trailing_nul_separator(self):
        # powermetrics separates successive samples with a NUL byte.
        raw = _plist(gpu={"idle_ratio": 0.5}) + b"\x00"
        assert metrics.parse_powermetrics(raw)["gpu_residency_pct"] == 50.0

    def test_accepts_str_input(self):
        raw = _plist(gpu={"idle_ratio": 0.25}).decode("utf-8")
        assert metrics.parse_powermetrics(raw)["gpu_residency_pct"] == 75.0

    def test_missing_fields_are_none(self):
        raw = _plist(gpu={}, processor={})
        assert metrics.parse_powermetrics(raw) == {"gpu_residency_pct": None, "ane_mw": None}

    def test_garbage_is_none(self):
        assert metrics.parse_powermetrics(b"not a plist at all") == {
            "gpu_residency_pct": None,
            "ane_mw": None,
        }


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
        assert "$@" not in metrics.WRAPPER_CONTENT
        assert "$*" not in metrics.WRAPPER_CONTENT
        assert "$1" not in metrics.WRAPPER_CONTENT

    def test_wrapper_requests_both_samplers_and_plist(self):
        # GPU residency is only in gpu_power; ANE power only in cpu_power;
        # plist gives machine-readable output (verified macOS 26, Task 13).
        assert "cpu_power" in metrics.WRAPPER_CONTENT
        assert "gpu_power" in metrics.WRAPPER_CONTENT
        assert "plist" in metrics.WRAPPER_CONTENT


class TestSamplePowermetrics:
    def test_invokes_wrapper_via_sudo_n(self):
        proc = MagicMock(returncode=0, stdout=REAL_SAMPLE)
        with patch("immich_accelerator.metrics.subprocess.run", return_value=proc) as run:
            result = metrics.sample_powermetrics()
        cmd = run.call_args[0][0]
        assert cmd[:2] == ["sudo", "-n"]
        assert cmd[2] == str(metrics.POWERMETRICS_WRAPPER)
        # bytes stdout must NOT be decoded by subprocess (text=True absent)
        assert run.call_args.kwargs.get("text") in (None, False)
        assert result is not None
        assert result["gpu_residency_pct"] == 10.27
        assert result["ane_mw"] == 0.0

    def test_nonzero_return_yields_none(self):
        proc = MagicMock(returncode=1, stdout=b"")
        with patch("immich_accelerator.metrics.subprocess.run", return_value=proc):
            assert metrics.sample_powermetrics() is None

    def test_oserror_yields_none(self):
        with patch("immich_accelerator.metrics.subprocess.run", side_effect=OSError):
            assert metrics.sample_powermetrics() is None
