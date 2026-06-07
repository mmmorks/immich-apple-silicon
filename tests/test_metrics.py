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

    def test_extracts_gpu_residency_hw_label(self):
        assert metrics.parse_powermetrics("GPU HW active residency:  37.50%")["gpu_residency_pct"] == 37.5

    def test_extracts_gpu_residency_plain_label(self):
        assert metrics.parse_powermetrics("GPU active residency:  41.0%")["gpu_residency_pct"] == 41.0

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
        assert "$@" not in metrics.WRAPPER_CONTENT
        assert "$*" not in metrics.WRAPPER_CONTENT
        assert "$1" not in metrics.WRAPPER_CONTENT


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
