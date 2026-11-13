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

The joint physical-pixel writer is now trained and terminal. Read
`references/joint_writer_physical_result.md`: all41,298 retained updates,
independent source/optimizer audit,124 qualifications,33 opaque boards and2,080
frozen readings are complete. Original frozen credits are970/1,040 versus
599/1,040, but development books91/108 and training-fixed QA90/101 miss the
unchanged90% per-group prerequisite. Do not resume this fit, repair readings,
select an intermediate head, relax the gate or launch its708 autonomous streams.
Its complete language remains unmeasured. A later pixel/blank-flag audit finds
15of17 uncredited selected development-book outputs empty, two nonempty, and
two other empty book outputs credited. Across2080readings,17blank flags disagree
with pixels and4emptyimageswerecredited. Preserve original readings/grades and
disclose these review errors; see `references/joint_writer_blank_flag_audit.md`.
The probability diagnosis is also complete; read
`references/joint_writer_blank_density_result.md`. All124 cases/12,597positions
are scored without optimization or new generation. Only1/14selected devbook
falseblanks and no selected trainfixedQA falseblank prefer the source image;
the search-only criterion fails. No blank-prefix argmax inconsistency is found.
The next intervention is selected in `references/joint_writer_context_learning_decision.md`:
train continuation adapters/query through the proper joint image likelihood,
with the original head-only trial as the unchanged control. Its exact state-VJP
implementation and real-source GPU preflight now pass; read
`references/joint_context_preflight_result.md`. All131 trainables match ordinary
autograd and both complete513-glyph book cases fit within17.795GiB. All updates
were reset; preserve the original SIGTERM and one traced recovery. Do not repeat
the completed preflight. The next full fit is defined in
`references/joint_context_learning_protocol.md`; commit/push its exact registration
before neural execution. Preserve all50 preflight sources, the original45 control
sources, and every original frozen reading. Use the new serial controller only
after checking live ownership; its full fit remains unproved until audited.
The post-qualification evaluator is now implemented in
`scripts/evaluate_joint_context_language.py`; see
`references/joint_context_language_execution.md`. It restores all131 learned
context/head tensors, preserves the full708stream plan and independently checks
that credited candidate images contain ink. CPU admission/restoration checks do
not establish full-body generation or language. Do not run it before complete
fit/audit/physical qualification and a committed post-gate execution binding.
The old head-only evaluator cannot substitute for this context-aware restore.
Use the private handoff for current ownership. The old point system's whole
7,578,390,003 parameters and the joint head's2,642,368 trainables do not establish
a lightweight foundation; whole-system costs remain required.

The saved-trajectory and initial/final clean-history studies are complete; read
`references/visual_continuation_trajectory_result.md` and
`references/continuation_clean_history_result.md`. Both long cache checks pass.
Final proportional development QA reaches F1 .95 on 656/713 later supplied-history
positions; sampled training books reach it on only about 9%. No autonomous score
changes. Do not repeat these completed controls or the completed joint fit.
Use their evidence for failure-specific decisions, with resource admission and
live ownership.

Update the Nature-style manuscript and native figures from the completed
checkpoint results. Existing PDFs predate this measurement; preserve their
baselines and label their status accurately. Model failure resolution retains
priority over presentation-only work. Use the private handoff for current
ownership and resources. The old trial is terminal; current joint-writer runtime
ownership belongs only in the private handoff.

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

