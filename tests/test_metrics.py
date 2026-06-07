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

    def test_parses_real_macos26_output(self):
        # Captured verbatim from `powermetrics --samplers cpu_power,gpu_power`
        # on Apple Silicon / macOS 26 (Mac17,9) during Task 13 verification.
        # GPU residency comes from the gpu_power block, ANE power from cpu_power.
        real = (
            "*** Sampled system activity (1006.08ms elapsed) ***\n"
            "\n**** GPU usage ****\n\n"
            "GPU HW active frequency: 338 MHz\n"
            "GPU HW active residency:  10.65% (338 MHz:  11% 486 MHz:   0%)\n"
            "GPU idle residency:  89.35%\n"
            "GPU Power: 119 mW\n\n"
            "CPU Power: 602 mW\n"
            "GPU Power: 103 mW\n"
            "ANE Power: 0 mW\n"
            "Combined Power (CPU + GPU + ANE): 705 mW\n"
        )
        result = metrics.parse_powermetrics(real)
        assert result["gpu_residency_pct"] == 10.65  # not idle/frequency lines
        assert result["ane_mw"] == 0.0


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

    def test_wrapper_requests_both_samplers(self):
        # GPU residency % is only in gpu_power; ANE power only in cpu_power.
        # Both are required (verified on macOS 26 / Mac17,9, Task 13).
        assert "cpu_power" in metrics.WRAPPER_CONTENT
        assert "gpu_power" in metrics.WRAPPER_CONTENT
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
