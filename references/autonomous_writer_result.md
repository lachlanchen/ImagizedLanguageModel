# Direct final-image learning improves copying but remains unreliable

The completed 32-update pilot gives **15/32 seen and 13/32 held-out OCR-exact
copies**, compared with 6/32 and 9/32 in its unchanged starting checkpoint.
It meets the prospective exploratory progress rule: at least three additional
held-out exact observations without aggregate seen regression. It does
**not** meet the 30/32 exact, legible requested strings alone required in each
split. Omissions, repeated characters, duplicate lines and copied instructions
remain visible. There is still no qualified usable ILM.

All 64 cases are retained in original order, with target, starting output and
final output shown vertically:
[seen 00–15](evidence/autonomous_writer_result/all_cases_00_15.png),
[seen 16–31](evidence/autonomous_writer_result/all_cases_16_31.png),
[held-out 00–15](evidence/autonomous_writer_result/all_cases_32_47.png),
[held-out 16–31](evidence/autonomous_writer_result/all_cases_48_63.png).
The [complete export](evidence/autonomous_writer_result/result.json) contains
every input, target, starting/final PNG, transcript, source report and the full
update journal. Each final PNG equals the quantized actual neural decoder
tensor. No font substitution, OCR repair or selected reroll is used. This
authenticates saved outputs, not an independent neural replay.

## Complete before/after observation

| Independent CPU OCR observation | Starting state | After 32 updates |
| --- | ---: | ---: |
| Seen, all strings | 6/32 | 15/32 |
| Held out, all strings | 9/32 | 13/32 |
| Seen Chinese | 1/16 | 3/16 |
| Seen English | 5/16 | 12/16 |
| Held-out Chinese | 2/16 | 3/16 |
| Held-out English | 7/16 | 10/16 |

The same evaluator reads 27/32 seen and 28/32 held-out clean references exactly.
OCR underrecognizes some legible generated writing; these counts are not manual
reading accuracy. Fifteen individual outputs change from OCR incorrect to
correct, while two change from correct to incorrect, for a net gain of thirteen
across both splits. The full export retains every change.

Some changes are visible content corrections. The first held-out English
phrase restores the missing `n` in `open`. Seen case 015 removes its extraneous
line and writes the requested English phrase. Seen case 030 restores a missing
Chinese character. Other changes concern typography or recognition, so the
OCR gain cannot be interpreted entirely as newly learned content.

Failures remain clear without relying on OCR: seen case 004 omits characters,
case 012 still copies part of the instruction, and case 024 retains an extra
form. Held-out case 000 adds a quotation mark, case 002 repeats a character,
case 026 duplicates its line, and case 030 copies the instruction. Held-out
case 028 introduces an opening quotation mark absent from the starting output.
More than two obvious failures per split independently reject writing admission.

## What was optimized

The [registered protocol](autonomous_writing_protocol.md) starts from the failed
recognition-treatment checkpoint, with the same externally pretrained
FLUX.2-klein-base-4B, VAE and constant generic condition. Each update generates
a complete page from independent noise and the input image, using 50 native
steps and guidance one. Targets enter only the final-image loss:

    L = CE(reader(native_display_pixels), target_string + EOS)
        + 10 * balanced_L1(unclipped_decoded_RGB, target_RGB).

The pixel term weights target ink and background means equally, addressing
visible errors the training reader can ignore. This tests a canonical writing
style, not font-independent or historical-script generalization. Gradients
traverse all 50 steps; no correct target mixture or target code enters sampling.
Native inference remains independent of the reader, labels and target images.
The method applies established [direct reward fine-tuning](https://arxiv.org/abs/2309.17400),
rather than introducing a new backpropagation algorithm.

All 32 sampled IDs and independent noise seeds reproduce the declared CPU RNG
streams. Sampling with replacement visits 22 distinct examples from the existing
32-example set; all had been seen by the starting model. Every update uses
FP32 AdamW, learning rate 1e-5, decay .01 and clipping one. The two preflight
updates are discarded before the pilot. The [native execution proof](autonomous_writer_preflight_result.md)
and [input-precision correction](autonomous_sampler_correction.md) remain separate.

| Measured invocation | Time (s) | Peak allocated GPU memory (GiB) |
| --- | ---: | ---: |
| 32 additional full-trajectory updates | 1,135.452 | 12.057 |
| Fresh 64-case native inference | 518.400 | 8.031 |

These times exclude preceding training, acquisition, preflight, separate CPU
evaluation and report generation. The deployed transformer has 3,899,903,488
parameters, including 24,358,912 locally trainable parameters, plus an
84,046,115-parameter VAE. The 560,528,640-parameter reader is training only.
External pretraining is retained. No small from-scratch model or efficiency
advantage at matched language quality is established.

## Interpretation and next requirement

This is one before/after continuation, not a matched comparison against another
32 updates of the previous objective. It changes the loss and uses corrected
native input-encoding precision. Extra optimization, the pixel term, recognition
supervision, numerical interface and sampling variation are not separately
isolated; there is no independent-seed replication. The gain supports another
controlled learning test under the prospective rule, not attribution to one
mechanism or unchanged large scale-up.

No natural QA is optimized or evaluated. Semantic understanding, general
instruction following, Chinese etymology, broader writing forms and the
independent general ILM objective remain unproved. Reliable autonomous writing
and meaningful unseen image-question answering are still required. Reproduce
with `scripts/train_autonomous_writer.py`, `scripts/infer_paired_image_cohort.py`,
`scripts/score_paired_writing.py` and `scripts/export_autonomous_writer_result.py`.
