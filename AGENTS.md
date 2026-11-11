# AGENTS.md

Scope: Entire repository tree.

Purpose: Operating conventions for agents and collaborators working in this repo.

Note on precedence: If direct system/developer/user instructions conflict with this file, follow those direct instructions first.

For model development and publication work, read the current acceptance criteria
and experiment decision in [the engineering goal](docs/first-imagized-language-model-goal.md).
Before starting another model experiment, identify the unresolved language
measurement it will address and check the current GPU owner and saved progress.
Prioritize completion and held-out evaluation of the registered full-sequence
experiment. Keep copying, codec fidelity, language quality and resource savings
as distinct claims. Record new evidence and the next decision in the goal and
private runtime handoff; keep the goal active while its acceptance criteria
remain unproved.

Both compact complete-sequence fits have failed the language criteria.
The registered 1,024-channel writer fit, five generations, all 372 opaque
readings and authenticated scoring are complete. Its rank repair gives no
complete exact QA improvement. Preserve both controls, their five checkpoints,
raw failures and source identities. The trained cold child passes the
recovered access/pixel audit; this is an execution result.

Before another training branch, follow the goal's failure-localization plan:
measure visual access, conditional writing under supplied reference history,
sampler sensitivity and emitted-history drift with frozen weights. Register
actual cases, matched noise, metric and resource limits before model calls.
Keep reference-supplied diagnostics separate from autonomous language scores.
Choose the next bounded intervention from these measurements. Reuse the
completed corpus and teacher cache unless measured evidence requires new data.

## Always Commit and Push After Change

Policy: After making any change to files in this repository (e.g., via `apply_patch` or manual edits), create an atomic commit and push it to the tracked branch.

Rationale: Keeps history auditable, enables easy rollback, and supports collaboration.

## Standard Workflow

1) Ensure repository is initialized
   - If not already a Git repo, run:
     - `git init`
     - `git branch -M main` (or keep existing default)

2) Stage changes
   - `git add -A`

3) Commit with a concise, conventional message
   - Format: `type(scope): summary`
   - Examples:
     - `docs(ilm): add imagized-language-model design draft`
     - `feat(raster): implement grid rasterization pipeline`
     - `fix(embed): correct level-3 code decoding`

4) Set remote and branch tracking (first push only)
   - If `origin` is not set, configure it (example):
     - `git remote add origin git@github.com:lachlanchen/ImagizedLanguageModel.git`
   - Ensure the working branch exists locally, e.g., `main`.

5) Push after every commit
   - First push: `git push -u origin <branch>`
   - Subsequent: `git push`

## Safety and Exceptions

- Do not embed credentials in commands. Rely on the environment/agent’s configured auth (SSH keys, tokens).
- If no remote is configured or network is unavailable, still commit locally; push when remote becomes available.
- Group logically related edits into a single atomic commit. Avoid mixing unrelated changes.
- Large/binary artifacts: prefer storing in `data/` and consider `.gitignore` or LFS if appropriate. Do not commit sensitive data.
- If a direct system/developer/user instruction forbids committing for a specific action, that instruction overrides this policy for that action.

## Commit Message Guidance

- Keep subject ≤ 72 characters; use imperative mood.
- Optional body with wrapped lines clarifying rationale and scope.
- Reference files or modules briefly rather than pasting large diffs.

## Branching

- Default branch: `main` (unless otherwise specified by the repository).
- For larger features, use short-lived feature branches and open PRs; merge via fast-forward or squash depending on project preference.

## Example Session

```
git add -A
git commit -m "docs(ilm): add imagized-language-model design draft"
git push
```

