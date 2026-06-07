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
