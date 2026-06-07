"""Apple Silicon hardware metrics via powermetrics.

powermetrics requires root. We do NOT run the dashboard as root; instead
a fixed, root-owned wrapper script is installed and granted a scoped
passwordless sudoers rule (see __main__._install_powermetrics_sudoers).
The wrapper hard-codes the invocation so the grant can't be abused with
arbitrary args.

``parse_powermetrics`` is a pure function over the wrapper's stdout so it
is unit-testable. ``sample_powermetrics`` runs the wrapper via ``sudo -n``
(non-interactive — fails instead of prompting if the rule is absent).

We use ``-f plist`` (machine-readable) rather than scraping the human
text output — the text labels and sampler layout vary across macOS
versions, but the plist keys are structured and stable. Verified on
Apple Silicon / macOS 26 (Mac17,9):
  - GPU active residency % is derived from ``gpu.idle_ratio`` (only the
    ``gpu_power`` sampler populates it): ``(1 - idle_ratio) * 100``.
  - ANE power (mW) is ``processor.ane_power`` (only the ``cpu_power``
    sampler populates it).
So the wrapper requests both samplers. ANE exposes power (mW), not a
utilization percentage.
"""

from __future__ import annotations

import math
import plistlib
import subprocess
from pathlib import Path
from typing import TypeGuard

POWERMETRICS_WRAPPER = Path("/usr/local/sbin/immich-accelerator-powermetrics")
POWERMETRICS_SUDOERS = Path("/etc/sudoers.d/immich-accelerator")

WRAPPER_CONTENT = "#!/bin/sh\nexec /usr/bin/powermetrics -n 1 -i 1000 --samplers cpu_power,gpu_power -f plist\n"


def sudoers_content(user: str) -> str:
    return f"{user} ALL=(root) NOPASSWD: {POWERMETRICS_WRAPPER}\n"


def _finite_number(x) -> TypeGuard[float]:
    """True only for a real, finite int/float.

    Excludes bools (``isinstance(True, int)`` is True) and NaN/inf —
    powermetrics can emit NaN on a cold sample, and a NaN serializes as the
    bare token ``NaN`` (invalid JSON), which freezes the dashboard's status
    feed. Such values must collapse to None instead.
    """
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def parse_powermetrics(raw: bytes | str) -> dict:
    """Parse one powermetrics plist sample into GPU residency % and ANE mW.

    Accepts the raw stdout (bytes preferred; str is encoded). powermetrics
    separates successive samples with a NUL byte, so we keep only the first.
    Returns None for any field that's absent or unparseable.
    """
    if isinstance(raw, str):
        raw = raw.encode("utf-8", "replace")
    raw = raw.split(b"\x00")[0]  # first sample only
    try:
        data = plistlib.loads(raw)
    except Exception:
        return {"gpu_residency_pct": None, "ane_mw": None}

    gpu = data.get("gpu") or {}
    proc = data.get("processor") or {}

    idle = gpu.get("idle_ratio")
    gpu_residency_pct = round((1.0 - idle) * 100, 2) if _finite_number(idle) else None

    ane = proc.get("ane_power")
    ane_mw = float(ane) if _finite_number(ane) else None

    return {"gpu_residency_pct": gpu_residency_pct, "ane_mw": ane_mw}


def sample_powermetrics() -> dict | None:
    """Run the privileged wrapper non-interactively; None if unavailable."""
    try:
        r = subprocess.run(
            ["sudo", "-n", str(POWERMETRICS_WRAPPER)],
            capture_output=True,  # bytes (no text=True) — plist may be binary
            timeout=8,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if r.returncode != 0:
        return None
    return parse_powermetrics(r.stdout)
