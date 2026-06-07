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
# leave it pinned to the default hooks dir (which holds only *.sample files).
# Unset it ONLY when it contains no real hooks; bail on a genuinely custom path
# so we never clobber an intentional hooks setup.
hooks_path="$(git config --get core.hooksPath || true)"
if [ -n "$hooks_path" ]; then
  resolved="${hooks_path/#\~/$HOME}"
  real_hooks="$(find "$resolved" -maxdepth 1 -type f ! -name '*.sample' 2>/dev/null || true)"
  if [ -z "$real_hooks" ]; then
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
