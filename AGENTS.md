# AGENTS.md

Scope: Entire repository tree.

Purpose: Operating conventions for agents and collaborators working in this repo.

Note on precedence: If direct system/developer/user instructions conflict with this file, follow those direct instructions first.

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

