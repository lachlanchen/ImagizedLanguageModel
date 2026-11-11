# AGENTS.md

Scope: Entire repository tree.

Purpose: Operating conventions for agents and collaborators working in this repo.

Note on precedence: If direct system/developer/user instructions conflict with this file, follow those direct instructions first.

For model development and publication work, read the current acceptance criteria
and experiment decision in [the engineering goal](docs/first-imagized-language-model-goal.md).
Before starting another model experiment, identify the unresolved language
measurement it will address and check the current GPU owner and saved progress.
Prioritize the current registered failure-localization measurement and held-out
language evaluation. Keep copying, codec fidelity, language quality and resource savings
as distinct claims. Record new evidence and the next decision in the goal and
private runtime handoff; keep the goal active while its acceptance criteria
remain unproved.

Both compact complete-sequence fits have failed the language criteria.
The registered 1,024-channel writer fit, five generations, all 372 opaque
readings and authenticated scoring are complete. Its rank repair gives no
complete exact QA improvement. Preserve both controls, their five checkpoints,
raw failures and source identities. The trained cold child passes the
recovered access/pixel audit; this is an execution result.

The frozen-weight diagnostic is complete for both final checkpoints. Correct
reference history and lengths still fail first-span writing; eight-to-64-step
sampling does not repair it. The image encoder's native-target alignment also
has a measured affine restriction. Use the goal and complete diagnostic receipt;
keep these supplied-reference results separate from autonomous language scores.

The earlier actor's current-cohort evaluation is complete: 2/32 strict exact
QA answers, 1/16 both-correct pairs, three readable QA outputs and no credited
complete book continuation. All 120 opaque outputs, authenticated scoring and
the fixed cold replay are complete. Its earlier 16/32 result uses another
cohort/metric. Preserve the current exact pair and the larger inherited-body
cost, alongside both compact failed controls.

The frozen full-prompt presentation comparison is complete: 6/32 strict exact
QA, 11/32 readable answers and 0/16 both-correct pairs, versus 2/32, 3/32 and
1/16 with fixed cells. One earlier exact answer becomes unreadable, so the
registered retention rule fails. All 96 outputs, scoring and fixed cold replay
are complete. Preserve both presentations and their unrepaired outputs.

Before another fit, finish the registered helper-only native-reader diagnosis in the
goal: the same full QA prompts and erased controls in both presentations,
with actual adapted versus zero-output source adapters and separate complete
text controls. Match adapted image prefills to the actual actor's saved state.
Keep the source lexical decoder outside the ILM; no helper answer or development
reference becomes training data. Finish this measurement before choosing a
training branch. The 448-call registration and CPU preparation are complete;
the serial execution has started, with quality results pending. Check the
private handoff and live process identity before resuming or starting work.
An observation timeout does not authorize restarting a live worker.
Reuse completed assets and inspect prior semantic-glyph-loop
results before repeating a failed approach. A failed strict canonical match
does not authorize repairing the reading or substituting a post-hoc metric.

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

