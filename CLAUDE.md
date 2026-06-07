# Claude Code Instructions

## Git workflow

- This repo (`mmmorks/immich-apple-silicon`) and its `ml` submodule (`mmmorks/immich-ml-metal`)
  are our forks — we own `main` in both. No external approval is needed to merge or push;
  follow the active profile below for *when* to commit/push (default: when asked).
- Work on branches for non-trivial changes; squash merge to main (one clean commit per release).
- Version bump + CHANGELOG entry required for every release to main.
- Tag releases as `vX.Y.Z` matching the VERSION file.
- Each fork has an `upstream` remote pointing at the original `epheterson/*` repo. To contribute
  a change back, open a PR against upstream (`gh pr create --repo epheterson/<repo>`); never push
  to `upstream` directly.

## Code style

- Python: type hints, f-strings, pathlib for paths.
- Keep it simple. No abstractions for one-time operations.
- The ffmpeg wrapper is bash — keep it minimal, no unnecessary forks.

## Testing

- Deploy to Mac Mini (`ssh macmini`) and verify before claiming anything works.
- Use Playwright for dashboard screenshots.
- Check processing progress via the Immich API, not assumptions.

## Immich compatibility

- Use jellyfin-ffmpeg (same as Docker). Don't patch Homebrew ffmpeg.
- The goal is identical output to Docker Immich wherever possible.
- Document every deviation in the "Known differences" README table.


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->
