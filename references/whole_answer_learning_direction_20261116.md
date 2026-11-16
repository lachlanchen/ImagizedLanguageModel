# Train and assess complete image answers

The project has not demonstrated a useful language model. The completed
continuation trial yields 7/32 strict proportional answers, 0/16 both-correct
question pairs and no credited coherent complete book continuation. The later
ranked writer makes many individual forms readable, but its failed whitespace
gate leaves autonomous language unmeasured. These are distinct outcomes.
[Complete language result](visual_continuation_learning_result.md),
[ranked-writer result](global_glyph_ranking_physical_result.md).

The [completed white-target comparison](white_target_physical_result.md) fails
both physical blank conditions: 12/252 false ink against a maximum of eleven and
54 false blanks against a maximum of 37. Its fit, audit, qualification and opaque
review are terminal. The implementation below qualifies no model; the
[failure decision](white_target_failure_decision.md) requires a trustworthy
complete-answer learning signal before any successor fit.

## What the failures support

The present training objective scores complete physical glyphs on correct
preceding images, adds blank/nonblank margins and a stopping loss, and updates
the continuation adapters/query and writer. It can learn conditional image
associations; it is inaccurate to say it receives no language signal. However,
improving its loss does not establish that the deployed process chooses a
correct complete answer. A readable wrong date is still a wrong answer.

The current model inherits 7.522B parameters and deploys 7.553B in total. Only
5.006M are trained in this comparison. The inherited native body and prompt
memory are frozen. Transferring its language behavior through a new visual
feedback/writing interface is a learning problem in its own right. The small
trainable count is not evidence of an efficient new foundation.

The actual manifest contains 9,004 QA and 4,762 book sequences. Per epoch it
contains 170,991 QA target positions and 1,865,256 book target positions, repeated
across three presentations. These are target exposures, not unique facts or
tokens, and the objective averages within sequences; the larger book position
count does not by itself imply larger aggregate book gradient weight. The
6.109M scheduled positions are not foundation-scale pretraining. Conversely,
limited exposure alone does not explain every failed interface experiment.

Correct-history controls already show writing failures before autonomous
feedback can accumulate. Thus exposure bias is **one testable intervention**,
not the established sole cause. Teacher-state regression, question-assignment
loss, recognition-guided writing, faithful codecs, a correlated image law,
and the released MIXAR baseline have all been tried. Repeating their names is
not a new remedy. The current writer is already a joint density; the
independent-pixel conditional-mean explanation does not describe this head.
[Correct-history control](continuation_clean_history_result.md),
[earlier alternatives](conditional_writing_decision_20261109.md),
[MIXAR release check](pixel_language_research_check_20261111.md).

## A specific remaining intervention

Add a training term for **complete sampled answers on their own emitted pixel
histories**, preserving the supervised objective as an anchor. Whole-answer
supervision avoids assigning a gold next glyph by ordinal position after the
model has already written a different prefix. The first controlled application
is fixed-font modern-Chinese QA with complete reference images. It does not
force a unique reference continuation for an open-ended book.

For prompt image X and sampled binary glyph images Y, define the stopped or
censored trajectory law

\[
 \log \pi_\theta(Y,z\mid X)=
 \sum_{t<T}\log(1-s_\theta(h_t))+
 \sum_{t<T}\sum_{j=1}^{128}
       \log q_\theta(b_{tj}\mid h_t,b_{t,<j})+
 z\log s_\theta(h_T),\qquad h_t=R_\theta(X,Y_{<t}).
\]

Here T is the emitted count, z=1 for a sampled terminal stop and z=0 for
right-censoring at a fixed cap. The cap contributes no invented stop action.
Physical eight-bit patterns b encode pixels, not characters or Unicode IDs.
Sampling uses the exact normalized law at temperature one. Greedy/top-k
sampling cannot be replayed as though it came from that same law.

For G independent answers to the same prompt, use a leave-one-out baseline:

\[
 A_i=r_i-\frac{1}{G-1}\sum_{k\ne i}r_k,\qquad
 L_{\rm policy}=-\frac1G\sum_i\operatorname{stopgrad}(A_i)
                    \log\pi_\theta(Y_i,z_i\mid X).
\]

With fixed reward and an unchanged policy between sampling and replay, this is
an unbiased score-function estimator of the negative expected reward gradient.
It sums all action log probabilities. Dividing by answer length, treating
greedy outputs as on-policy samples, or differentiating a changing reward
would change that statement. Equal group rewards give zero signal and must
be reported, not counted as successful language training. This uses established
REINFORCE/RLOO methodology rather than a new optimization theorem.
[Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740).

Related pixel-language work trains on free-running trajectories. MIXAR's
Chinese free-generation scores remain weak, and its release was already tested
here. Its result motivates this control without proving our candidate or
consumer-GPU foundation training.
[MIXAR v2, sections 3.3 and 4.3](https://arxiv.org/html/2604.11575v2).

## Implemented component and its limits

[`on_policy_glyph_learning.py`](../ilm/visual_lm/on_policy_glyph_learning.py)
provides ancestral sampling from the existing 128-piece physical writer,
sampled stopping, actual emitted-image feedback, differentiable replay and
memory-serial group backward. It does not retain several backbone graphs.
References enter the reward only. No OCR, lexical head, source-glyph selection
or font renderer enters sampling. The existing inference code is unchanged.

The initial fixed-font surrogate is an ordered image edit score with
foreground-Dice substitution and a complete exact-image bonus. It penalizes
insertions, deletions and duplicate glyphs, preserves real spaces, and gives
zero to capped or wholly blank answers. It is **not a validated semantic
reward**: a wrong similar-looking glyph can earn partial credit, and legitimate
alternative wording or writing styles can lose credit. The exact bonus does
not prevent all reward exploitation. Frozen recognition guidance failed
earlier in part because the recognizer tolerated visibly wrong outputs;
replacing that judge requires calibration, not another assumption of fidelity.
[Earlier recognition failure](autonomous_writing_loss_result.md).

Seventeen CPU checks pass, including exact enumeration of the RLOO expected
gradient, sampled-versus-dense probability replay using the actual production
GlyphJointDensity architecture, finite gradients through its parameters,
random-generator replay, stop/censoring accounting, and adversarial reward
fixtures for blank, reordered, duplicated and similar-but-wrong glyphs. The
density fixture has random weights. These are implementation controls, not a
trained model result. Full inherited-body sampling/replay, resource use,
optimizer replay and language benefit remain unexecuted. The saved-output
calibration below rejects the first reward. No neural successor fit is
registered or running.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 \
  python -m unittest tests.test_on_policy_glyph_learning -v
```

## Completed reward falsification

The [preregistered CPU comparison](evidence/whole_answer_reward_calibration_registration_20261116.json)
binds 235 inputs and all 64 final normal QA outputs, 32 in each presentation,
from the completed original continuation trial. It reads original packed
output pixels and authenticated reference images from the prepared corpus;
it loads no model, teacher alignment or checkpoint and renders no new glyphs.
The implementation and registration were committed and pushed before scoring.
Frozen transcriptions and uncertainty remain unchanged.

The [result](evidence/whole_answer_reward_calibration_20261116.json) contains
9 strict complete-correct answers, 9 readable wrong answers and 46 outputs
without a credited clear reading. Reward separates the first two groups
(pairwise AUC 1.0; minimum correct 0.485744, maximum readable-wrong 0.452647).
However, one proportional output with an uncertain frozen reading receives
0.492818, above the minimum correct reward. The prospective requirement that
every correct answer outrank every uncredited output therefore fails. All
capped outputs receive zero. The decision is **reject this reward for policy
optimization**. Do not tune the gate or repair the uncertain reading.

This result does not establish that the uncertain output is semantically wrong;
it establishes that the reward lacks the required conservative separation on
this reused, model-reviewed cohort. There is no independent human adjudication.
It neither qualifies a replacement semantic judge nor measures an ILM learning
gain. Scoring takes 0.539 seconds at 0.631 GiB peak process RSS; imports and
source authentication precede the scoring timer. There are no new model calls
or optimizer updates. An independent scalar recount confirms 64 rows, 9 exact,
18 readable, 46 uncertain and all capped rewards zero.

The remaining candidate needs a trustworthy **training-only** semantic judge
before a fit. One possible contract is to transcribe only generated answer
images, abstain on ambiguous writing, and compare the frozen transcript with
the training answer in a separate matcher. Do not show the question or reference
to the transcriber. Require agreement on visible glyph count, order, repeated
glyphs, true spaces and stopping, including adversarial duplicate/missing-stroke
cases. Earlier recognition losses did not satisfy that contract, so a new
recognizer is not presumed to do so. Validate it on preserved real outputs
before using its reward; a failed judge stops the policy proposal. Such a
judge is absent from deployed generation. Its preparation and runtime costs
must still be counted. No training-only judge has been qualified.

The subsequent [training-only selective content verifier](selective_visual_verifier_protocol_20261116.md)
is now implemented and tested. It rejects two of nine known-correct complete
answers and accepts one uncertain output, failing its registered qualification.
It is not a qualified replacement judge. Both rejected rewards remain preserved
for historical reproduction. The generic policy helper now refuses a missing
or non-callable explicit reward before model prefill or sampling; qualification
of a future callback remains the registered caller's responsibility. Nineteen
policy-helper and seven selective-verifier CPU checks pass. No successor
student training or full inherited-body policy replay has run.

The next fixed qualification is the [isolated-cell vision reader](isolated_glyph_judge_protocol_20261116.md).
It transcribes physical cells independently, retaining their count and order,
then freezes those readings before a separate answer matcher uses reference
strings. The registered cohort retains all 64 earlier outputs and every clean
reference cell. No new student fit is admitted by its implementation; the fixed
reader must first pass its conservative reused-output calibration.

## Execution decision

The original serial white-target comparison is complete and fails its physical
prerequisite. Preserve all 115 bound sources and the first frozen review. Its
708-stream language comparison remains inadmissible; no execution binding exists.
Do not restart the completed fit, run extra epochs or substitute an intermediate
checkpoint. No successor fit is admitted by this result.

The first reward failed calibration; do not run its policy fit or model
support preflight. Before registering any replacement whole-answer fit,
calibrate its fixed reward on actual preserved output images and frozen
readings, including readable wrong answers,
wrong dates/entities, duplication, whitespace, early stops and caps. Preserve
all uncertain readings. Use disagreement to reject the reward; never repair
old readings from references. Register a small **training-only** support and
gradient preflight on a fresh clone, with a fixed sampled cap, before any
model calls. If samples are uniformly unreadable or rewards do not discriminate
correctness, stop this route before spending a full training budget.

Keep the full supervised corpus and the unchanged parent as controls. Any
policy schedule must state which QA sources receive the additional term and
account for their additional exposure; it may not claim a matched optimization
budget if sampling adds compute. Default helper caps are not an experiment
schedule. Five training answers exceed 128 glyphs and 1,650 exceed 32, so a
short cap cannot silently stand for the complete corpus. Choose a fixed cap
and report all censoring prospectively. Do not change the existing evaluation
caps or gates.

A GPU preflight needs the existing GPU0 mutex, at least 48 GiB available host
RAM at entry and 24 GiB at runtime, at most 22 GiB allocated VRAM and 48 GiB
free disk. Bound total samples, elapsed time and replay error before launch;
retain all outputs and discard a pending update on any replay mismatch. Measure
rollout cost before approving a full schedule. Sequential sampling may be too
expensive or high-variance to justify this intervention on one 4090.

The fixed acceptance evidence remains complete correct answers, both questions
in each passage pair, erased-input controls, learned stopping, retention and
coherent book output. Evaluate the unchanged greedy deployment separately from
the sampled training policy; do not report best-of-group answers as autonomous
success. Reject the intervention if reward improves without those language
gains. Only after a complete reference model works should a smaller student
inherit it and receive separate full-quality and resource comparisons.

Representing writing as images removes a fixed lexical output inventory. It
does not itself create semantic knowledge or reduce the computation needed to
learn it. This intervention tests a missing training connection between
language intent and actual writing; its effectiveness remains unproved.
