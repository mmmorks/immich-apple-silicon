# Claude Code Instructions

## Git workflow

- This repo (`mmmorks/immich-apple-silicon`) and its `ml` submodule (`mmmorks/immich-ml-metal`)
  are our forks — we own `main` in both. No external approval is needed to merge or push;
  this repo's active agent profile is `team-maintainer` — see "Active agent profile" below
  for *when* to commit/push.
- Commit directly to `main` — do not create feature branches.
- After committing changes inside the `ml` submodule, bump the submodule pointer in
  this parent repo (`git add ml && git commit`) so the parent records the new `ml` SHA.
  A submodule commit alone leaves the parent pointing at the old revision.
- **Keep `ml` in line with its own `origin/main` — don't trust the recorded pointer.**
  The parent's recorded `ml` gitlink can *lag* `ml`'s `origin/main` (a bump may never have
  been committed/pushed), so a plain `git submodule update` leaves you building/testing
  stale `ml` code. Before starting work **and** before any commit/push, sync the submodule
  to its origin and re-bump the pointer if it moved:

  ```bash
  git -C ml fetch origin main
  git -C ml checkout -B main origin/main            # ml working tree == ml's origin/main tip
  git add ml                                        # stage the pointer bump (commit per the active profile)
  # verify they match — this should print nothing:
  test "$(git -C ml rev-parse HEAD)" = "$(git -C ml rev-parse origin/main)" || echo "ml STILL not at origin/main"
  ```

  Only `checkout -B main origin/main` *after* any local `ml` commits are pushed (step 3),
  or you'll move the branch off unpushed work.
- Version bump + CHANGELOG entry required for every release to main.
- Tag releases as `vX.Y.Z` matching the VERSION file.
- Each fork has an `upstream` remote pointing at the original `epheterson/*` repo. To contribute
  a change back, open a PR against upstream (`gh pr create --repo epheterson/<repo>`); never push
  to `upstream` directly.

## Active agent profile

This repository **explicitly opts into the `team-maintainer` profile** (defined under
*Agent Context Profiles* in the Beads block below). That block's `Conservative (default)`
wording is the fallback for repos that have *not* chosen a profile — it does **not** govern
this repo. Treat `team-maintainer` as the active policy: agents may run quality gates,
commit, push, and `bd close` as part of finishing work without asking first. A live
"do not commit" / "do not push" instruction from the user always wins.

## Closing out bead work

Under the team-maintainer profile, when working a bead, **drive it to completion**: run the
full commit/push workflow (per "Git workflow" / "Working in a git worktree" above) and
`bd close` it once the acceptance gates pass — don't stop to ask "should I commit this?".
If the *only* open question is whether to commit/push and all acceptance criteria + quality
gates (tests, lint, build) are green, proceed without asking. Pause for human input only on
genuine obstacles: failing or ambiguous gates, design decisions, destructive/irreversible
actions, or anything the bead's acceptance criteria don't cover.

## Working in a git worktree

We sometimes start a session inside an isolated worktree (launched with `claude -w`).
The worktree branch is a **scratch copy of `main`** — commits land on `origin/main`
directly; the branch never gets its own `origin/<branch>` upstream.

**Detect it at session start.** You are in a linked worktree (not the primary checkout) when:

```bash
[ "$(git rev-parse --git-dir)" != "$(git rev-parse --git-common-dir)" ] \
  && ! git rev-parse --show-superproject-working-tree 2>/dev/null | grep -q .
# true ⇒ linked worktree; the superproject check excludes false positives inside the ml submodule
```

When that holds, follow this flow (otherwise use the normal `main` checkout flow above):

1. **Initialize the submodule — required first step.** `git worktree add` does **not**
   populate submodules, so `ml/` starts empty and any build/test touching it fails
   confusingly. Run once at session start:

   ```bash
   git submodule update --init --recursive            # populate ml/ at the parent's recorded SHA
   git fetch origin main && git rebase origin/main     # parent: start from the current tip of main

   # The recorded ml pointer can lag ml's own origin/main — bring the submodule
   # in line with ITS origin too, or you build/test stale ml code (see Git workflow above):
   git -C ml fetch origin main
   git -C ml checkout -B main origin/main              # ml now at its own origin/main tip
   git add ml                                          # stage the pointer bump if it moved (commit per profile)
   ```

   The `ml` submodule shares its object store with the primary checkout, so this is cheap.
   Sanity-check before doing any work: `git -C ml log --oneline -1` should show `ml`'s
   `origin/main` tip, not an older SHA.

2. **Mid-session: commit locally, do not push.** The branch has no upstream and we don't
   want one. The `ml`-pointer-bump rule from the Git workflow above still applies: commit
   inside `ml`, then `git add ml && git commit` in the parent.

3. **When committing/pushing (per the active profile or when asked): push to `main`, then
   re-sync the worktree.** If you changed `ml`, push the submodule first so the pointer the
   parent records is reachable on the `ml` remote:

   ```bash
   # only if ml changed:
   git -C ml push origin HEAD:main

   # parent: land the worktree's commits on main, then bring the branch back in sync
   git fetch origin main && git rebase origin/main
   git push origin HEAD:main
   git fetch origin main && git rebase origin/main   # worktree HEAD now == origin/main

   # re-sync ml to its origin too, so the worktree's ml isn't left behind:
   git -C ml fetch origin main && git -C ml checkout -B main origin/main
   ```

   Resolve any conflicts in the rebase; **never force-push to `main`**. After this both the
   worktree branch and its `ml` submodule match `origin/main` — the worktree is back in sync.

## Code style

- Python: type hints, f-strings, pathlib for paths.
- Keep it simple. No abstractions for one-time operations.
- The ffmpeg wrapper is bash — keep it minimal, no unnecessary forks.

## Keep bead IDs out of code, comments, docs, and commits

Bead/issue IDs (`ml-1s2`, `ml-7j8.14`, `immich-xxx`, etc.) are tracker bookkeeping.
They mean nothing to someone reading a clone of this repo, so **anything that ships
in the tree or in git history must be self-contained** — explain the *what* and *why*
directly, never by pointing at a bead.

- **Code, comments, docstrings, READMEs, and committed docs/plans:** describe the
  behavior, bug, or rationale in plain words. Don't append `(ml-xxx)` tags and don't
  use a bead ID as a noun (write "after the open_clip fallback was removed", not
  "after ml-b82"; "the wrong-weights fix", not "ml-7j8.17").
- **Commit messages:** write a standalone summary of the change. A trailing
  `(ml-xxx)` reference is tolerated as metadata, but the message must make full sense
  with it stripped out — never let the bead ID carry meaning the message omits.
- **Where bead IDs belong:** the `bd` tracker, `bd` commands, and ephemeral session
  chatter only — not the source tree.
- This applies to the `ml` submodule too (it has no `CLAUDE.md` of its own; this rule
  governs work there). When in doubt, read the line as a stranger would: if dropping
  the ID loses information, you denormalized too little; if the ID was the only thing
  there, you forgot to write the actual explanation.

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

**Architecture in one line:** `bd` data lives in a shared Dolt sql-server (machine-wide; see global `~/.claude/CLAUDE.md`), NOT a per-repo embedded DB. The `ml-*` issues live in the shared *planning* DB hydrated as an additional repo — not this repo's `.beads` — so `bd export`/`bd stats` run here can report 0; use `bd list`. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md.

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
