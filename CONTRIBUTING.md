# Contributing

## First-time setup

Install the local git hooks once per clone (or worktree):

```bash
./scripts/setup-git-hooks.sh
```

This wires up [pre-commit](https://pre-commit.com): **ruff** (lint + format),
a **gitleaks** secret scan, and the fast **pytest** suite all run automatically
on `git commit` (and the same ruff + gitleaks hooks for the `ml/` submodule).
The gitleaks hook is the real defense against committing a credential — it scans
staged changes *before* they leave your machine. The CI `secrets-scan` job only
runs after a push, by which point a leaked secret is already on the remote and
must be rotated.

Requires `pre-commit` (`pipx install pre-commit`) and `gitleaks`
(`brew install gitleaks`) on your PATH.

## Before you push

```bash
pytest -v -m "not slow"
```

All tests must pass. CI runs the same suite (plus ruff, pyright, and gitleaks)
— if it fails there, the PR won't be merged. With the hooks installed (see
above), ruff/gitleaks/pytest already run on every commit.

## What to test

- `pytest` covers compose template validation, regex patterns, config parsing, and fresh-install regressions
- If your change touches the ffmpeg wrapper, dashboard, or worker startup, test on a real Mac with the accelerator running
- ML changes go through the [upstream repo](https://github.com/sebastianfredette/immich-ml-metal)

## PR guidelines

- One concern per PR
- Keep diffs small — under 200 lines is ideal
- Update CHANGELOG.md if user-facing
