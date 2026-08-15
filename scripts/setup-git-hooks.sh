#!/usr/bin/env bash
#
# Set up local git pre-commit hooks for this repo and the ml/ submodule.
#
# Installs the hooks defined in .pre-commit-config.yaml (ruff lint + format,
# a gitleaks secret scan, and the fast pytest suite). Run this once per clone
# or worktree:
#
#     ./scripts/setup-git-hooks.sh
#
# Why it matters: the gitleaks hook scans staged changes BEFORE they're
# committed, so a leaked credential never reaches the remote. The CI
# secrets-scan job only runs after a push — by then a secret is already public
# and must be rotated. Local hooks are the actual prevention.
#
# Requires: pre-commit (pipx install pre-commit / brew install pre-commit)
#           gitleaks  (brew install gitleaks)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# --- tool availability -------------------------------------------------------
if ! command -v pre-commit >/dev/null 2>&1; then
  echo "ERROR: pre-commit not found on PATH." >&2
  echo "  Install it with:  pipx install pre-commit   (or: brew install pre-commit)" >&2
  exit 1
fi
if ! command -v gitleaks >/dev/null 2>&1; then
  echo "WARNING: gitleaks not found on PATH — the secret-scan hook will fail to run." >&2
  echo "  Install it with:  brew install gitleaks" >&2
fi

# --- clear a redundant core.hooksPath ----------------------------------------
# pre-commit refuses to install when core.hooksPath is set. Worktree setups can
# leave it pinned to the repo's own default hooks dir. Unset it when it is
# redundant — either it resolves to the default hooks dir (where hooks would be
# read from anyway, so unsetting changes nothing) or it holds no real hooks.
# Bail only on a genuinely custom path, so we never clobber an intentional setup.
hooks_path="$(git config --get core.hooksPath || true)"
if [ -n "$hooks_path" ]; then
  resolved="${hooks_path/#\~/$HOME}"
  # NOTE: `git rev-parse --git-path hooks` honours core.hooksPath, so it would
  # echo the pin straight back and every path would compare equal. The common
  # git dir ignores core.hooksPath, and is where hooks live by default (it is
  # also shared across linked worktrees).
  default_hooks="$(git rev-parse --git-common-dir)/hooks"
  # Compare canonical paths: a hooksPath equal to the default dir is a no-op
  # pin, even once pre-commit has installed a real hook into it.
  same_dir=""
  if [ -d "$resolved" ] && [ -d "$default_hooks" ] \
     && [ "$(cd "$resolved" && pwd -P)" = "$(cd "$default_hooks" && pwd -P)" ]; then
    same_dir=1
  fi
  real_hooks="$(find "$resolved" -maxdepth 1 -type f ! -name '*.sample' 2>/dev/null || true)"
  if [ -n "$same_dir" ] || [ -z "$real_hooks" ]; then
    echo "Unsetting redundant core.hooksPath ($hooks_path) so pre-commit can manage hooks…"
    git config --unset-all core.hooksPath
  else
    echo "ERROR: core.hooksPath points to a custom hooks dir with existing hooks:" >&2
    echo "  $resolved" >&2
    echo "  pre-commit will not install over it. Resolve this manually, then re-run." >&2
    exit 1
  fi
fi

# --- install hooks -----------------------------------------------------------
echo "Installing pre-commit hooks for the parent repo…"
pre-commit install

if [ -f ml/.pre-commit-config.yaml ]; then
  echo "Installing pre-commit hooks for the ml/ submodule…"
  ( cd ml && pre-commit install )
fi

echo
echo "Done. Active hooks: ruff (lint + format), gitleaks (secret scan), pytest (fast)."
echo "Run them across all files at any time with:  pre-commit run --all-files"
