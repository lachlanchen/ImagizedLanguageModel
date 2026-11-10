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

## Next controlled intervention

The [genuine copy curriculum](../references/genuine_glyph_readout_protocol.md)
uses8,172training images covering4,086visible forms under two instructions,
plus256held-wording images. One empty form is explicitly excluded. Each target
bitmap appears only as part of the image shown to the frozen reader; no class
label enters its forward pass. Batch-versus-single states must pass their
numerical check before caching continues.

The existing G will train on these genuine states mixed with the original
language histories, with E,D,stop and body fixed. Only after its physical
qualification and actual image review can it enter the full40-stream autonomous
language assessment. This experiment is in progress; no result is claimed.
A CPU supervisor waits for the exact cache process to exit, then runs fitting
and qualification sequentially. It stops before image review and never retries
failed stages or weakens their gates.

[Failure localization](../references/pixel_language_source_reader_result.md) ·
[Motor diagnosis](../references/semantic_glyph_motor_diagnosis.md) ·
[Decoder test](../references/robust_glyph_decoder_result.md) ·
[Dense flow](../references/semantic_motor_flow_result.md) ·
[Spatial flow](../references/spatial_semantic_motor_result.md) ·
[Rank calculation](../references/glyph_readout_rank_result.md) ·
[Current manuscript](../publication/ilm-nature/manuscript.pdf)
