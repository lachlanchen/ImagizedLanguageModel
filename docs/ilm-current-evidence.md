# Current ILM evidence and next test

The verified runtime reads a question image, emits32x32glyph images and encodes
its actual emitted pixels before continuation. It has no runtime tokenizer,
lexical output head, font renderer, character prototype table or teacher
service. It retains an externally pretrained7.5Bvisual/contextual body and a
47.1Mneural bridge; this is not a small foundation model trained from scratch.

The [standalone image interface](semantic-glyph-language.md) is cold-qualified
without training-corpus/font access or network inference. Its strongest
completed language evidence remains restricted date composition:11of14new date
conditions on two selected question templates have clear matching output-first
transcriptions, all stop naturally, and three damaged/wrong outputs remain
failures. Manual readings are non-independent. Broader Chinese assessment still
has11capped streams out of16and only one canonical-exact full answer.

## What caused the earlier failures?

The original larger writer fits495/512complete training answers but0/24normal
development answers. A matched diagnostic source reader recovers literal
references on12/16Chinese questions using exactly the same image-memory
fingerprints. This localizes substantial lost capability to writing; it does
not uniquely separate query, history, head and capacity effects.

The complete-glyph loop improves the interface but has sparse target coverage:
1,074positive forms and266initial forms in508supported questions, compared with
4,087primitive image forms. A fixed-state check finds an allowed reference
first form nearest in semantic cosine on9/16normal questions, while only1/16
first images is pixel-exact. Nearest-anchor ranks are evaluator diagnostics;
they never replace generated output.

A [coverage join](../references/glyph_target_coverage_result.md) puts eleven of
those sixteen cases in the group with an allowed first form already seen at
an initial training position. All nine nearest-code successes and the one
exact first image occur there. Two other cases have a form seen only later
in training answers; three have no form seen as a language target. All five
lack a first image reaching F1 0.95. These are small descriptive strata, not
evidence that coverage alone causes the failures. Known-form motor errors and
transfer to new initial targets both need testing.

The trained semantic output span has uniform anchor-regression residual0.30518.
The best possible rank2048affine span could reach0.060109. This identifies a
measurable latent fitting limit and a poor fit of the current span to the wider
inventory. It does not prove that nonlinear image decoding requires full rank.

## Completed attempts that did not qualify

| Intervention | Actual physical result | Decision |
| --- | --- | --- |
| Synthetic hidden-state readout compiler | F1 0.523 synthetic / 0.830 genuine; genuine quality regresses | Reject |
| Added contrastive readout loss | F1 0.490 / 0.795 on those same pools | Reject |
| Decoder noise training | Clean F1 0.99975; random-noise 0.99205; hard mixtures 0.82275 | Mixed-condition gate fails |
| 168.4M dense motor flow | Clean F1 0.90628; mixtures 0.71545 | Reject |
| 3.89M spatial image flow | Clean F1 0.99826; mixtures 0.81054 | Clean generation works; combined gate fails |

The first two rows measure teacher-associated glyphs on calibration/genuine
state pools. The remaining rows measure canonical forms under their respective
fixed corruption protocols. These are different tasks/schedules, not a
matched-compute model ranking or semantic accuracy. All raw outputs, including
wrong forms and sampled candidates, are retained in the linked result reports.

Sharp known-form image generation is therefore feasible on one4090D, but it
is insufficient to resolve an uncertain semantic code. Repeated neural E/D
passes do not improve the five normal first images reaching0.95F1. More motor
architectures alone have not supplied a reliable language model.

## What about image-feedback drift?

An [audit of existing native-head labels](../references/cached_glyph_teacher_agreement_result.md)
on all4,363gold-history training states finds next-form agreement72.44%at the
first glyph and78.64%across continuation glyphs. Agreement dips to64.69%after
one glyph, then reaches84.85%after two or three. Later positions represent
different question subsets; the report includes their matched first-state
baselines. This is neither an autonomous answer test nor a causal comparison
against exact-source feedback. It does not support assuming that cumulative
image-feedback damage alone explains the failures. The audit uses previously
cached labels, makes no neural call, and leaves the current experiment intact.

## Genuine contexts and the direct-pixel result

The [genuine copy curriculum](../references/genuine_glyph_readout_protocol.md)
uses 8,172 training images covering 4,086 visible forms under two instructions,
plus 256 held-wording images. One empty form is explicitly excluded. Each target
bitmap appears only as part of the image shown to the frozen reader; no class
label enters its forward pass. Batch-versus-single states must pass their
numerical check before caching continues.

The [completed fits](../references/genuine_pixel_readout_result.md) show strong
copying improvement. With the anchor losses, held-wording mean F1 reaches
0.951778, but only 194/256 cases reach F1 0.95, missing the declared 80% gate.
The same G trained with pixel loss only reaches mean F1 0.972856 and 229/256
cases (89.45%), retaining original-history quality. Both use the same architecture
and 18,000-update schedule; E, D, stop and body stay fixed. All raw glyphs are
archived and all three sampled boards for each candidate were actually viewed.

The pixel-only candidate passes the unchanged physical gate. Its completed
forty-stream assessment retains all eight training controls, but yields zero
exact canonical development images, mean foreground F1 0.341013 and eleven
capped streams out of sixteen. The prior baseline has one exact image, F1
0.354426 and the same eleven caps. Better copying has not improved complete
answering. All forty raw outputs were viewed before current reference scoring;
[every stream and score is retained](../references/evidence/direct_pixel_language_g_only_v1/archive.json).
Its extra-capacity comparison did not train because shared
memory pressure exceeded that arm's entry limit; the failed admission remains
preserved. The CPU identity audit's 252/256 held retrieval result is evaluator
geometry, not a runtime glyph bank or a generated answer.

A [separate C3 reading cohort](../references/c3_visual_reading_preparation.md)
was prepared before this run's qualification: 16 passages, 64 image inputs, with
original/rotated options and letter/full-phrase instructions. All input preview
boards were inspected. Its [completed frozen-checkpoint assessment](../references/c3_glyph_language_result.md)
has zero correct complete phrases out of 32 for either checkpoint. The candidate
gives eleven conservative clear letter matches out of 32, with only three
documents correct under both option orders. These agent readings have no
independent human grade; damaged and ambiguous letters remain uncertain.
Every raw output is retained. This is not general language competence, and
inherited-body pretraining exposure is unknown.

The [matched feedback diagnostic](../references/glyph_feedback_ceiling_protocol.md)
compares native source units, exact character input vectors and learned
canonical-image input vectors with a temporary complete source head. It makes
no autonomous ILM images. Its purpose is to distinguish state-to-glyph writing,
character segmentation and feedback alignment before further fitting. No C3
input is used, and diagnostic trajectories are excluded from training. The
[completed result](../references/glyph_feedback_ceiling_result.md) gives 12/16
literal matches with native units versus 3/16 with either character route.
All 72 initial states match exactly. Source units can span multiple characters;
substituting one character per native step does not preserve the source policy.
This does not prove that its hidden state cannot support a separately trained
glyph policy, and encoder error remains possible.

The [next training candidate](../references/glyph_tail_adaptation_protocol.md)
therefore trains the final four contextual attention layers together with G and
stop on 3,176 supported training questions. Its lower 32 layers stay frozen and
their causal prefix states can be cached. Two real preflight updates use 3.208
GiB peak allocation, change 294,912 adapter-up coordinates and retain all frozen
parameters. A protocol layer-count correction is explicitly preserved; the
implementation selected the actual final four layers throughout. This preflight
is a compute/gradient check, not a language-learning result. The complete fit and
autonomous assessment remain required.

[Failure localization](../references/pixel_language_source_reader_result.md) ·
[Motor diagnosis](../references/semantic_glyph_motor_diagnosis.md) ·
[Decoder test](../references/robust_glyph_decoder_result.md) ·
[Dense flow](../references/semantic_motor_flow_result.md) ·
[Spatial flow](../references/spatial_semantic_motor_result.md) ·
[Rank calculation](../references/glyph_readout_rank_result.md) ·
[Current manuscript](../publication/ilm-nature/manuscript.pdf)
