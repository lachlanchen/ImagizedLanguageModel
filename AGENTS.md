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

The full continuation-only trial is complete: 13,766 sequences, 41,298 updates,
6,108,741 scheduled target positions and all 732 prescribed output streams.
All 183 opaque boards and 96 visible normal-book prefixes have actual review
receipts; readings were frozen before new mapping/reference scoring. The final
model gives 7/32 strict proportional answers, 0/16 both-correct pairs, 14/32
readable answers and no credited coherent complete long-book continuation.
Both the original scorer and independent four-QA-erasure audit reject the pilot.
Read `references/visual_continuation_learning_result.md` and
`references/visual_continuation_failure_decision.md` before further model work.

Training, generation, review and optimizer/source audits are terminal. Do not
restart them, launch obsolete evaluator recovery, or repeat unchanged earlier
native-reader, writer or correct-history diagnoses. Preserve all failures,
fixed checkpoints, original scores and uncertain readings. No source weights,
reference answers or images may be used to repair frozen transcriptions.

The next intervention is not selected or registered. Use newly frozen complete
trajectories to distinguish initial writing, later collapse, question/passage
use and stopping. A new neural diagnostic must name a changed checkpoint or
unresolved contrast and its decision value. Register one bounded intervention
and an unchanged control before another fit. More unchanged epochs, selecting
an intermediate checkpoint or relaxing acceptance rules do not address the
failed final pilot. The whole deployed model is 7,578,390,003 parameters;
20,212,225 trainables do not establish a lightweight foundation.

Update the Nature-style manuscript and native figures from the completed
checkpoint results. Existing PDFs predate this measurement; preserve their
baselines and label their status accurately. Model failure resolution retains
priority over presentation-only work. Use the private handoff for current
ownership and resources; no model worker is live after this completed trial.

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

