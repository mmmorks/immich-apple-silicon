# ML Appliance Mode — Design

**Date:** 2026-06-06
**Status:** Approved (design); implementation pending
**Branch/target:** `main` on fork `mmmorks/immich-apple-silicon` (`upstream` = `epheterson/immich-apple-silicon`)

## Problem

A NAS runs full Immich (server + microservices worker) and offloads machine
learning to a remote endpoint over HTTP (`IMMICH_MACHINE_LEARNING_URL`). Today
that endpoint is the stock `immich-machine-learning` Docker container running on
a Mac — which is **CPU-only**, because Docker on macOS cannot reach the Metal
GPU or Neural Engine.

This repo already contains a native, Metal/ANE-accelerated ML service (the
`ml/` git submodule → `immich-ml-metal`). It speaks the **exact same HTTP API**
as the Docker ML container (`/ping`, `/health`, `POST /predict`) and binds
`0.0.0.0:3003` by default. But the CLI bundles it with the native microservices
worker: `immich-accelerator start` always starts the worker too, and `setup` is
built entirely around the worker topology (shared filesystem, matching absolute
paths, Postgres/Redis, server extraction, `/build` link, ffmpeg wrapper).

There is no supported way to run **just** the GPU-accelerated ML service as a
drop-in remote ML endpoint for a separate Immich host.

## Goal

Add a first-class **ML appliance mode**: the Mac runs only the native Metal ML
service as a remote ML endpoint. No Docker, no worker, no shared filesystem, no
database access on the Mac. `setup`, `start`, `watch`, `status`, `stop`, `logs`,
`dashboard`, and `uninstall` all support this mode.

## Key facts (verified)

- ML service routes: `GET /`, `GET /ping` (→ `pong`), `GET /health` (model load
  state), `POST /predict` (multipart: `entries` JSON + optional `image`/`text`).
  Handles CLIP visual/textual, face detect+embed, OCR. — Same API as the Docker
  ML container.
- Binding env vars: `ML_HOST` (default `0.0.0.0`), `ML_PORT` (default `3003`).
  → LAN-reachable out of the box.
- Logging: `ML_LOG_LEVEL` (default `INFO`), `ML_LOG_REQUESTS` (default `true`).
  Service is launched via `start_service("ml", ...)`, so it logs to
  `~/.immich-accelerator/logs/ml.log` → per-request throughput parsing is viable
  with no upstream change.
- ML inference is **stateless over HTTP** — it ships image bytes in and returns
  embeddings/JSON. It never reads the photo library on disk. Hence no shared
  filesystem is required (unlike the worker).
- The ML service has **no authentication** (Immich ML never has).
- The existing dashboard's "Metal GPU / Neural Engine" bars are **placeholders**
  driven by queue activity (`dashboard.html:320-323`), not real counters. There
  is no `powermetrics`/`ioreg` usage anywhere in the repo today.
- `powermetrics` requires **root**; there is no entitlement that lets an
  unprivileged user run it.

## Decisions

1. **First-class mode**, not a config hack — supported across all relevant CLI
   commands.
2. **Pure ML appliance** — the Mac depends on nothing but Python + the ML venv.
   No Docker/DB/Redis/worker/server/shared filesystem.
3. **ML-focused dashboard** — adapt the dashboard to ML health, throughput, and
   real Apple-Silicon utilization; drop worker/queue panels.
4. **Real GPU/ANE metrics via `powermetrics`**, using a **scoped passwordless
   sudoers rule** that points at a fixed, root-owned wrapper script (not
   `sudo powermetrics` with user-controllable args). The dashboard process stays
   unprivileged.

## Architecture

Add a `mode` field to config: `"full"` (existing default) or `"ml-only"`. Each
existing command gets an early branch on `mode`. Branch-in-place (not a parallel
command tree) so the venv setup, PID/log/`start_service` plumbing, and launchd
machinery are reused. The ml-only paths skip all Docker / DB / Redis / worker /
server-extraction / media-path / `/build` / ffmpeg logic.

### Config schema (ml-only)

Minimal — no `db_*`, `redis_*`, `server_dir`, `upload_mount`, `api_key`:

```json
{
  "mode": "ml-only",
  "ml_dir": "<repo>/ml",
  "ml_host": "0.0.0.0",
  "ml_port": 3003,
  "metrics_powermetrics": true,
  "dashboard_port": 8420
}
```

`load_config`/`save_config` unchanged; `mode` defaults to `"full"` when absent
(back-compat for existing installs).

## Components

### 1. `setup --ml-only` → `_setup_ml_only()`

Lean setup path:

1. Ensure Python 3.11+; create the `ml/` submodule venv + install
   `requirements.txt` (reuse `_find_ml_dir`).
2. Prompt for ML port (default 3003) and bind host (default `0.0.0.0`).
3. Install the powermetrics sudoers grant (Component 5) — single sudo prompt.
4. Offer the launchd service (watch mode, ml-only).
5. Write the ml-only config.
6. Print **NAS wiring**: the exact line
   `IMMICH_MACHINE_LEARNING_URL=http://<mac-LAN-ip>:3003` (Mac LAN IP detected
   best-effort via `ipconfig getifaddr en0`/`en1`; print candidates), and the
   instruction to stop the Docker `immich-machine-learning` container.

New CLI flag: `setup --ml-only` (action=store_true). `cmd_setup` branches to
`_setup_ml_only()` when set.

### 2. `start` → `_start_ml_only()`

`cmd_start` early-returns into `_start_ml_only()` when `mode=="ml-only"`:

- `_kill_stale_processes()` (ML only).
- Ensure ML venv (reuse `_find_ml_dir`).
- Launch `python -m src.main` via `start_service("ml", ...)` with env including
  `ML_HOST`/`ML_PORT` from config.
- Start the dashboard (Component 4).
- Write PIDs.

None of the worker preflight (Docker detection, `IMMICH_WORKERS_INCLUDE`,
media-path validation, server extraction, `/build`, ffmpeg wrapper) runs.

### 3. `watch` / `status` / `stop` / `logs`

- `watch`: ml-only branch monitors ML (+ dashboard) only, restarts on crash.
- `status`: ml-only branch shows ML health, host:port, throughput, powermetrics;
  no worker/queue.
- `stop`: stop `ml` + `dashboard` (no worker expected).
- `logs`: in ml-only, default to `ml`; `logs worker` errors cleanly.
- `ml-test`: unchanged — already hits `localhost:ml_port`.

### 4. ML-focused dashboard

Factor `get_status()` into `get_status_full()` (existing behavior) and
`get_status_ml(config)`, sharing a `_system_metrics()` helper (load avg, mem,
cpu count). The ml variant drops DB counts, queue API, worker RSS, and the
requeue endpoint, and adds:

- **Health**: `/ping` + `/health` (per-model load state).
- **Throughput/latency**: new pure function `parse_ml_log(text, window_s)` →
  per-task req/s and p50 latency, bucketed CLIP→GPU panel, faces/OCR→ANE panel.
  Reads `~/.immich-accelerator/logs/ml.log`. **The exact request-line format
  must be captured from a real log on the Mac Mini and the parser written to
  match it** (kept as a pure function over a fixture string for unit testing).
- **GPU/ANE**: real numbers from Component 5.

`dashboard.html` reads `status.mode` from `/api/status` and renders the
appliance layout (real GPU residency %, ANE power mW, throughput) instead of the
worker/queue/video panels. The `/api/requeue` endpoint is disabled in ml-only.

### 5. powermetrics via scoped sudoers

Secure realization of "sudoers rule" — avoids `sudo powermetrics <user args>`
(arg injection → root escalation):

- **Root-owned, fixed-arg wrapper** installed at
  `/usr/local/sbin/immich-accelerator-powermetrics` (`root:wheel`, mode 0755):

  ```sh
  #!/bin/sh
  exec /usr/bin/powermetrics -n 1 -i 1000 --samplers gpu_power
  ```

- **Sudoers drop-in** `/etc/sudoers.d/immich-accelerator` (mode 0440):

  ```
  <user> ALL=(root) NOPASSWD: /usr/local/sbin/immich-accelerator-powermetrics
  ```

  Validate with `visudo -cf <tmpfile>` **before** moving into place; abort on
  failure.

- New `metrics.py`:
  - `sample_powermetrics()` runs the wrapper via `sudo` and returns parsed data.
  - `parse_powermetrics(text)` — pure, unit-testable — extracts **GPU active
    residency %** and **ANE power (mW)**. ANE is power-only (no utilization %):
    surface as "active + N mW".

- `uninstall` removes the wrapper and the sudoers drop-in.

**To verify on the Mac Mini:** the exact `--samplers` needed for ANE power
(whether `gpu_power` includes the "ANE Power" line on this hardware/macOS, or
whether `cpu_power` must be added), and the real text layout the parser keys on.
`powermetrics -i 1000` adds ~1s latency per sample → acceptable because
dashboard status is cached (`_CACHE_TTL`).

### 6. Security posture (documented)

The ML service has no authentication and binds `0.0.0.0` — any host on the LAN
can call `/predict`. Document this in the README appliance section. Offer
`ml_host = <specific LAN IP>` as a narrowing option for users who want to bind a
single interface.

## Testing

### pytest

- `parse_powermetrics(text)` against a captured fixture (GPU residency + ANE mW).
- `parse_ml_log(text, window_s)` against a captured `ml.log` fixture (req/s, p50,
  per-task bucketing).
- Mode dispatch: `start`/`setup`/`watch` route to the ml-only paths when
  `mode=="ml-only"` and to the full paths otherwise.
- ml-only config write produces the minimal schema (no db/redis/server keys).
- Sudoers content + `visudo -cf` validation; wrapper file mode/ownership.

### Mac Mini E2E (per CLAUDE.md — deploy and verify before claiming it works)

1. `immich-accelerator setup --ml-only`.
2. ML service reachable on LAN `:3003` from another host.
3. Real `POST /predict` returns embeddings, running on Metal.
4. Point the NAS `IMMICH_MACHINE_LEARNING_URL` at the Mac, stop the Docker ML
   container, confirm Smart Search / faces / OCR jobs process.
5. Dashboard shows real GPU residency under load.
6. `immich-accelerator ml-test` passes.
7. launchd `watch` restarts the ML service after a kill.

## Docs / release

- README: new "ML appliance (remote ML endpoint for a NAS)" section, the
  security note, and a "Known differences" row.
- Release chores (VERSION bump, CHANGELOG, tag, Homebrew tap) handled at merge
  time per CLAUDE.md — not part of this work.

## Out of scope (YAGNI)

- Moving the microservices worker to the Mac / shared filesystem.
- Queue control or "Run All Missing" from the appliance dashboard.
- An upstream `/metrics` endpoint on the ML service.
- Per-ANE utilization percentage (hardware exposes power, not %).
