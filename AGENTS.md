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

The native diagnosis is complete: all 448 calls and 192 actual adapted image
prefills are verified. Adapted proportional native decoding gives 10/32 exact
answers versus 6/32 glyph answers. Source-adapter-zero proportional native
decoding gives 13/32; adapter effects are mixed across layouts and text.
Both erasures remain zero exact in every image arm. The helper lexical decoder
stays outside the ILM and does not qualify its image output. The worker exited.

The CPU first-writing/coverage audit is complete: all 192 original streams,
745 target positions and 128 displayed tiles pass the independent recount.
All first forms are covered; supplied reconstruction is 31/32 pixel-exact.
Pixel mismatch is not character/semantic error, and complete language scores
remain unchanged. Do not repeat this audit as pending work.

The actual-context conditional motor is implemented, registered and fitted:
24.478M parameters,12,000 verified updates in 109.485s at 0.989 GiB peak allocated
GPU memory. Its independent CPU optimizer audit passes. Keep the zero-update
CUBLAS admission failure and successful same-recipe retry. Do not refit or
interpret finite loss as language. Its 216 streams, 54 opaque boards and 12 book
prefixes are now reviewed and independently recounted: fixed 2/32 exact,
1/16 pairs, 4/32 readable; proportional 7/32 exact,1/16 pairs, 14/32 readable.
No complete book continuation is credited. The prospective pilot fails and
one of six prior exact proportional answers is lost. All 216 initial reader
states match the parent bit-for-bit. The fixed standalone replay passes after
removing old training-ledger reads from standalone admission; preserve the
original failed-access attempt and its exact pixels. The goal remains active.
Localize initial writing, generated-history drift and stopping from the saved
failures before another fit. Preserve the reader, adapters,
encoder, decoder, stop predictor and unchanged complete-response controls.
Earlier synthetic semantic flows and balanced deterministic fitting already
failed relevant gates; a new flow is a hypothesis, not a successful solution.
No first-image metric can substitute for readable complete answers, paired
question dependence, book continuation and untouched confirmation. A failed
strict canonical match does not authorize repairing the reading or changing
the metric. Verify live process identity before resuming work; an observation
timeout does not authorize a duplicate worker.

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

