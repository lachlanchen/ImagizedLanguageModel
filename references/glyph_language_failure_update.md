# Better glyph copying has not solved language generation

The original image-feedback loop still gives only restricted language evidence.
The genuine-copy curriculum improves physical writing; its pixel-only variant
passes the unchanged qualification, but the complete development answers remain
unreliable. This rules out using a copying score as the acceptance criterion
for the language model. It does not rule out image-based language learning.

The [complete results](genuine_pixel_readout_result.md) retain the matched
baselines, all images and failures. The selected year-edit intervention from
the original model remains evidence of limited input-dependent composition,
not general language competence. Neither a new physical theory nor a more
elaborate decoder has yet explained or fixed the complete-answer failure.

## What the objective actually trains

Let the frozen reader/body state be

\[
h_t = F(V(x), E(y_1), \ldots, E(y_{t-1})),\qquad
\hat y_t = B(D(G(h_t))).
\]

Here \(x\) is the prompt image, \(y_t\) a whole glyph image, \(E\) the image
feedback encoder, \(G\) the contextual readout, \(D\) the image decoder and
\(B\) its fixed binary threshold. Global affine denormalization of \(E\) is
suppressed in this notation. The learned stopping head also sees \(h_t\).

The initial fit uses reference-image histories. Its empirical objective is
therefore evaluated on states induced by the reference images, while runtime
states depend on the actual emitted images. Even perfect empirical fitting
does not guarantee correct decisions on newly encountered states. Moreover,
the inherited body was pretrained with source units that may contain several
characters; one glyph step is not generally one original language step.

The motor code need not equal the next feedback code. Because continuation
re-encodes the actual image, the necessary requirement is on the composite
\(D\circ G\), not equality between \(G(h_t)\) and a source character embedding.
Removing that extra equality requirement improves held-wording copying from
F1 0.951778 to 0.972856 with the same architecture and training schedule. The
complete-answer F1 nevertheless changes from the preserved baseline's 0.354426
to 0.341013. These are distinct conditioning distributions and measurements.

The original 508-question fit provides only 266 distinct initial target forms.
All five development cases whose initial reference form was absent from the
initial targets fail the original nearest-form and exact-pixel tests; familiar
initial forms can also fail. This small descriptive join does not establish
that more data alone will solve the problem. The broader copying data cover
glyph identities but do not supply varied language decisions for those forms.

## The next decision must follow a causal comparison

The [fixed feedback comparison](glyph_feedback_ceiling_protocol.md) holds the
prompt prefill, body and source decision head constant, then intervenes on the
kind of feedback. Native-unit versus exact-character feedback probes the step
size mismatch. Exact-character versus image-encoded feedback probes the learned
feedback interface. Trajectories may diverge, so this is a whole-policy
intervention rather than an estimate of one local encoder perturbation.

If the source policy can answer under canonical-image feedback, a useful next
training dataset consists of its genuine training-question states paired with
their next glyph images and terminal actions. That would target the actual
state-to-writing function instead of requiring a readout to infer it from a
small collection of reference strings and unrelated copy contexts. A later
rollout correction stage could address student-induced states. Both proposals
require evaluation; neither has been implemented or demonstrated by this note.
If the oracle itself fails under image feedback, writer-only distillation is
insufficient and the feedback interface needs attention first.

The dependence of future states on previous predictions is the setting studied
by [Ross, Gordon and Bagnell (2011)](https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf).
Its assumptions and expert guarantees are not established here. Pixel-language
generation and its readability difficulties also predate this project:
[PIXAR](https://arxiv.org/abs/2401.03321) explicitly studies them. These papers
motivate concrete controls; they provide no evidence that this implementation
already meets the general ILM objective.

The fresh C3 cohort was fixed before qualification. Both checkpoint identities
are frozen for that assessment, and no subsequent diagnosis or training may
turn its results into model-selection data. Success must mean useful complete
image answers on unseen questions, with measured stopping and prompt dependence,
not a new loss name, a clearer isolated glyph or a teacher rendering the answer.
