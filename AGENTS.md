# AGENTS.md

Scope: Entire repository tree.

Purpose: Operating conventions for agents and collaborators working in this repo.

Note on precedence: If direct system/developer/user instructions conflict with this file, follow those direct instructions first.

## Model-work priority

Read [the engineering goal](docs/first-imagized-language-model-goal.md) and the
private `artifacts/glyph_tail_current_handoff.json` at the start of each model
work unit. Use the goal for current acceptance criteria and the private handoff
for verified live ownership. Historical result files are evidence, not current
scheduling instructions.

The compact fits, writer comparisons, native-reader diagnosis, correct-image-
history comparison and latest realized-writer publication are complete. The
usable-language goal remains unachieved. Both correct-history physical gates
fail; reference history does not qualify autonomous language. The next direction
is complete-sequence training and evaluation of the continuation-only model.
The real all 36-layer preflight passes at 18.123 GiB with complete 513-glyph targets,
bit-exact frozen prompt memory and independently checked four-step optimizer.
Read `references/visual_continuation_preflight_result.md`. Do not repeat the
completed preflight or reuse its four updates for the full fit. Prepare and
register the substantial-corpus comparison before training, with complete QA
and book targets, both layouts, fixed snapshots and original language controls.
The full pipeline is implemented; indexing has authenticated 13,766 eligible
sequences and 2,036,247 target positions per epoch. Finish natural rendering and
registration, then execute `scripts/run_visual_continuation.py` in one owned
session. Read `references/visual_continuation_full_protocol.md`; its three epochs,
five snapshots, 732 output streams and final pilot are fixed before training.
Verify the real first-six-update resume check. Do not rerun indexing or launch
another worker merely because an observation handle has expired.
Preserve the failed mixtures, first-function transfer and writer-only controls.
Do not prioritize manuscript polish over the unresolved model failure.

Keep copying, glyph fidelity, complete language, stopping, execution independence
and resource savings as distinct claims. Preserve original outputs, controls and
successful cases. Every new experiment needs an observed failure, one bounded
change, fixed evaluation and resource ceilings, and an action for either result.
Update the goal and private handoff when the decision changes. Keep the session
goal active until its acceptance criteria are proved. Verify ownership before
launching a job; an observation timeout does not authorize another worker.

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

