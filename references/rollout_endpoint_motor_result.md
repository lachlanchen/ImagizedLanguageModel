# Full-sampler supervision improves first writing but fails the later gate

The registered full-rollout objective passes the fixed first-position writing
gate and fails the later-position gate. Its whole-answer quality therefore
remains unmeasured. This is partial physical improvement on fixed training
states, not a usable language model or a generalization result.

Both arms start from the same completed original-flow EMA and make 2,000
additional AdamW updates. They use identical actual-context targets, sampled
indices, Gaussian noise, interpolation times, optimizer schedule and EMA.
The only objective addition is balanced decoded BCE at the endpoint reached
by the existing 32-interval sampler. Its gradient traverses every interval,
using recomputation rather than a detached prefix. The reader, contextual
adapters, image encoder/decoder, normalization and stopping stay frozen.

| 32 training positions per pool | Original flow median F1 | Matched extra-FM median F1 | Full-rollout median F1 | Matched / rollout F1 ≥0.95 | Matched / rollout exact bitmaps |
|---|---:|---:|---:|---:|---:|
| First | 0.986408 | 0.986350 | 0.990581 | 32 / 31 | 5 / 6 |
| Later | 0.974518 | 0.977090 | 0.978614 | 24 / 25 | 0 / 2 |

Each pool requires median F1 at least 0.99 and at least 30/32 tiles at F1 at
least 0.95. The candidate passes first only; the matched control fails both.
The candidate's later mean F1 is 0.961013 versus 0.954906 for the matched
control. Supplied reconstruction remains 31/32 first and 32/32 later exact.
Residual changed components and broken strokes remain in the complete boards.
The first-position gain also loses one tile at the 0.95 threshold. This small
single-seed training-state comparison does not establish population effects.

All 64 normal development first states are retained with the original seeds.
Candidate canonical mean F1 is 0.462759 fixed-cell and 0.667170 proportional,
versus 0.448891 and 0.660603 for the matched control. Both new arms have zero
pixel-exact canonical first tiles in either layout. These are supplied-state
single-image measurements. Alternative answer prefixes and learned stroke
variation remain possible; canonical pixel difference does not grade meaning.
Stopping is unchanged, including the previously identified later nonterminal
state at probability 0.99542. The diagnostic emits one glyph regardless of
that decision. No complete language trial or new exact-QA score is assigned.

Seven numerical/runtime tests pass, including an actual-network finite
difference, bit-exact recomputed gradients and gradient transmission through
a frozen decoder. Before fitting, both GPU arms verify the differentiable
forward against the deployed sampler on actual training states. Independent
CPU audit checks all thirty parameter optimizer states at update 2,000,
finite model/EMA/moments, previous update-1,750 checkpoints and unchanged
normalization. Sampled-index digests and final CPU/CUDA random states match
bit-for-bit. Each arm samples 32,000 first and 32,000 later exposures. No new
head, lexical decoder, font renderer, answer bank or inference teacher enters
the actor.

The matched fit takes 103.775 seconds, including 80.072 seconds waiting for
memory pressure, at 1.165 GiB peak allocated GPU memory. The full-rollout fit
takes 875.962 seconds, including 80.073 seconds waiting, at 1.166 GiB. Both
complete within their fixed limits. Assessment through scoring takes 39.623 seconds at
0.402 GiB, with no pressure wait; terminal wall time including board export
is 40.027 seconds. Preflight hashing, independent CPU recount and visual
review are additional unmeasured wall costs. No full reader is loaded for these stages.
The inherited 7.522B visual/contextual body, earlier fitting and offline cache
construction remain excluded costs. No task-efficiency advantage follows.
The NVML inventory warning does not interrupt CUDA computation; actual
optimizer and output audits complete.

Every pixel/motor score is independently recounted from saved files, and all
512 displayed original tiles match their raw arrays. All eight boards are
viewed in a labelled model review with prior cohort familiarity, without
independent human adjudication. Raw process/session arguments remain in the
private assessment receipt. Public evidence retains aggregate resource and
identity bindings.

The full solver loss is more faithful to deployed endpoint computation than
the previous local estimator, but is insufficient under the registered gate.
This does not identify the sole remaining mechanism or reject all pathwise
training. Do not extend this run, change its coefficient, choose raw weights
or select another sampling seed. The existing complete-answer result remains
2/32 fixed and 7/32 proportional for the original flow, with failed retention.

Before another writer fit, compare the already physically qualified balanced
writer on this same cohort. Its earlier 18/32 clear-answer comparison uses a
different cohort and metric and cannot substitute for a matched result here.
This frozen control can distinguish a limitation of the new flow writer from
remaining reading, stopping and feedback failures. It requires a separate
prospective decision, complete outputs and unchanged scoring. No improvement
from that unrun comparison is claimed. General Chinese language, book
continuation, historical transfer and matched-quality efficiency remain open.

See the [decision and math](rollout_endpoint_motor_decision.md),
[registration](evidence/rollout_endpoint_motor_registration_20261111.json),
[independent fit audit](evidence/rollout_endpoint_motor_fit_20261111.json) and
[full assessment](evidence/rollout_endpoint_motor_learning_20261111.json).
The [pixel archive](evidence/rollout_endpoint_motor_20261111/pixels.npz) and
boards [1](evidence/rollout_endpoint_motor_20261111/board_01.png),
[2](evidence/rollout_endpoint_motor_20261111/board_02.png),
[3](evidence/rollout_endpoint_motor_20261111/board_03.png),
[4](evidence/rollout_endpoint_motor_20261111/board_04.png),
[5](evidence/rollout_endpoint_motor_20261111/board_05.png),
[6](evidence/rollout_endpoint_motor_20261111/board_06.png),
[7](evidence/rollout_endpoint_motor_20261111/board_07.png) and
[8](evidence/rollout_endpoint_motor_20261111/board_08.png)
preserve all positions without repair. The current verified manuscript
now includes this result and the subsequent matched balanced-writer failure
in the verified 35/62/63-page RealizedWriting revision. The earlier 33/58/59-page
MotorDiagnostics package remains preserved. See the
[completed balanced comparison](balanced_writer_matched_result.md) and
[revision record](manuscript_realized_writer_revision_plan.md).
