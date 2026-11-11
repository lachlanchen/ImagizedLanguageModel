# Next intervention: learn a glyph distribution from actual contextual states

## Evidence and hypothesis

The complete [native-reader diagnosis](glyph_tail_native_reader_result.md)
finds retained answer access in the inherited reader: adapted proportional
native decoding gives 10/32 exact answers, versus 6/32 actual glyph answers.
The [first-writing audit](glyph_tail_first_writing_result.md) finds all first
forms covered and near-perfect supplied-image reconstruction. Some original
first forms are already damaged before generated feedback. Other strict
mismatches concern wording, aliases or later output. Reading and stopping
remain imperfect. A better image motor alone cannot guarantee language.

Two earlier interventions limit the next choice. The
[balanced writer](balanced_glyph_writer_result.md) raises training first-form
F1 to 0.999559 without improving answers in new contexts. More deterministic
fitting on the same states is not an adequate sole remedy. The
[dense semantic flow](semantic_motor_flow_result.md) and
[spatial semantic motor](spatial_semantic_motor_result.md) fail uncertain
synthetic semantic mixtures, despite good clean-form rendering in the latter.
Both condition on an already predicted semantic code. Do not rerun those
primitive/mixture experiments as if they tested real contextual language.
The two compact joint fits also fail; their wider output projection and longer
sampling do not repair writing.

The next candidate tests a narrower hypothesis: predicting one deterministic
semantic point can lose a sharp glyph choice; a conditional distribution
trained directly on **actual frozen-reader states** may retain it. This is a
testable proposal, not a demonstrated mechanism or a promised repair. Sparse
state coverage and poor contextual generalization remain competing causes.

## One bounded change

Preserve the current 7.522B inherited reader, learned tail adapters, image
encoder, neural image decoder and stop predictor. Replace only the path from
contextual state to the decoder's continuous 64-by-4-by-4 motor feature with a
conditional flow head. Condition directly on the full 4,096-dimensional state,
not on the old uncertain semantic point or synthetic anchor mixture. Keep the
old deterministic writer as the unchanged control. The architecture/objective
replacement is one candidate package; this comparison will not isolate each
internal design choice causally.

For a training image I, let the frozen target be

    z = decoder_input(E(I)) in R^(64 x 4 x 4).

Estimate motor-coordinate mean and scale using training images only. With
standardized target y, independent Gaussian epsilon, and uniform t in [0,1],
fit

    x_t = (1-t) epsilon + t y
    L = E ||v_theta(x_t,t,h) - (y-epsilon)||^2 /1024.

The conditional optimal vector field is an expectation given both the noisy
motor state and context. Under the flow-matching assumptions it transports a
noise distribution toward a conditional endpoint distribution. This differs
from choosing one conditional mean image; finite training need not learn the
transport or choose the right language. See the primary
[Flow Matching paper](https://arxiv.org/abs/2210.02747). Continuous conditional
image distributions are also used by
[Li et al.](https://arxiv.org/abs/2406.11838); neither paper establishes success
for this ILM or supports a novelty claim.

Use a full-coordinate output path without the old narrow fixed velocity
projection. Capacity checks are execution checks only. The exact topology,
seed, optimizer, update count, normalization and numerical schedule must be
source-bound before fitting. Freeze one final-checkpoint sampler with one
sample per glyph and a fixed per-case seed. No best-of selection, glyph bank,
reference search, lexical decoder or answer-dependent stopping may enter
inference. Decode the sampled motor feature with the existing neural decoder;
feed back only its actual emitted binary pixels through the existing encoder.

## Reuse authentic training data; exclude diagnostic answers

Reuse the authenticated fixed-reader cache at
`artifacts/balanced_glyph_writer_v1/cache/pools.pt`: 3,176 first states and
20,086 later reference-conditioned states from the parent's 3,176 training
questions. Its report binds SHA-256
`201db406d906a2a1eaff916c14d24e256e77c24e9140c51c6e73cb42ab6aab5c`.
The cache reproduced all parent training pixel and stop scores. Its states are
actual reader states, unlike the older synthetic semantic mixtures. The
3,176 terminal states remain available for diagnostics, but stop weights stay
frozen in this intervention. Never train on this audit's 32 answers, first
states, native helper output or future confirmation references.

Authenticate every cache/label/source binding before reuse. Balance first and
later training positions without removing hard targets. This is a bounded
writer intervention on short QA, not broad book pretraining. The substantial
13,785-sequence book/QA corpus and its teacher cache remain preserved for a
later registered expansion. Do not silently use teacher-unit states as if
they were actual glyph-history states. Training-only representation coverage
and supplied reconstruction are diagnostics, not admission as usable language.

## Evaluation and resource decision

Evaluate the final candidate exactly once on the unchanged current cohort:
all 32 QA prompts and both erased conditions under both complete input
presentations (96 QA streams each), plus the twelve original fixed-cell book
prefixes and their twelve erased controls: 216 streams total. Retain the original stop threshold and
128-glyph cap; supply no target length. Keep original control outputs and
report every new complete stream, including failures and caps. Review raw
outputs before reference scoring, disclose prior cohort familiarity, and
apply unchanged strict and readability rules. Recount scores independently.

A development pilot passes only if proportional strict matches reach at least
8/32, both-correct pairs reach at least 2/16, readable answers reach at least
11/32, all six earlier exact proportional answers are retained, and both
earlier exact fixed-cell answers remain exact. Report erased-control scores,
book coherence, natural stopping, latency and resource use regardless of the
pilot outcome. These prospectively chosen development thresholds do not
qualify a usable model. Any pass requires a separate frozen evaluation on
untouched source groups, and the full language/continuation criteria remain.
Do not select a checkpoint or sampling seed from this reused cohort.

Before a fit, bind implementation, parent/cache hashes, training exposure,
sampler and full evaluation in a committed registration. Use GPU 0 under
`artifacts/ilm_gpu0.lock`, with one live project model job, at least 48 GiB
available host RAM at admission, 24 GiB during execution, at least 20 GiB free
GPU memory for the full reader, and a 22 GiB peak-allocation ceiling. Bound the
small-head fit to 1,800 wall seconds and each full-reader generation stage to
3,600 seconds, including at most 600 seconds pressure waiting; use the
existing PSI limits and preserve interrupted attempts. Do not duplicate the
reader or disturb another project's GPU service.

On failure, retain the unchanged actor and all failed streams. Do not prolong
the same fit or rerun more sampler steps without a new measured reason. On
pass, confirm independently before broader training. Continue to report the
inherited-body cost; a small head is neither a small independent base model
nor evidence of parity with a large LLM. A manuscript update should show the
actual trajectory, uncertainty and unrepaired output images.

Status: decision recorded after the complete CPU audit. This specific
actual-context motor-head intervention is not yet implemented, registered,
trained or evaluated. Earlier semantic-flow source files are preserved and
must not be mistaken for its implementation.
