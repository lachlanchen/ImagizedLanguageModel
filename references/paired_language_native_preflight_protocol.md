# Native full-page language preflight

The ILM goal requires answers that change appropriately with the visible
question. Random-string copying does not test that capability. The existing
CMRC-derived set has 256 training questions and 32 article-disjoint held-out
questions, paired by passage with disjoint accepted answers. This protocol
specifies the next geometry and execution preflight before a language-learning
experiment. It does not redefine success as passing a preflight.

No additional GPU stage may overlap the active ink-mass comparison. Its
authoritative process must first finish or fail, and its full result must be
preserved. Preparation of indices, pixel controls and CPU reference OCR does
not change that comparison's source, schedule, weights or evaluation.

## Fixed data and controls

Use the complete source pages in `artifacts/image_edit_learning/pages_v2` and
the preparation from `scripts/prepare_paired_language_controls.py`. All 288
natural-question inputs reproduce their original pixels and geometry exactly.
Passages remain complete, at 1024×1024 with the existing 32-pixel font.
Answers use the existing separate 512×256 pages. There is no answer-aware crop
or target-dependent input condition.

Evaluation normal inputs comprise the first 32 training questions in manifest
order and all 32 held-out questions. Both questions about one passage use the
same initial noise seed. Two held-out controls are prepared for every question:

- **Question erased:** retain every passage pixel, remove the question, and
  retain the fixed `问题：` and `回答：` labels. Normalize the latter label's
  vertical position to the longer question within each pair. Both controls in
  a pair are therefore bit-identical and use the same seed. This removes the
  question-length cue as well as its characters. The label position can change
  relative to a shorter normal question; this is explicitly not a perfectly
  layout-preserving content intervention.
- **Passage erased:** retain every question/answer-label pixel at its original
  position and the fixed `资料：` label; erase the passage content. This can
  leave geometric cues and cannot rule out answers from external pretraining.

Input-only indices contain only ID, input path/hash and seed. No answer,
question transcript, reference image or varying text embedding is accepted
by the deployed generator. Normal, erased-question and erased-passage indices
are separate so original IDs and seeds remain bound to their cases.

## Fixed first native checks

Use the complete balanced writer checkpoint
`fc031796540a4b6ca4c86774a5033e94ae32b415e825b6a221128756748d3575`
for this preflight, regardless of the ink-mass comparison's outcome. This fixes
geometry checks before any QA output is observed; it does not select the
starting point of a later full learning experiment.

The preflight normal cohort is the first training pair and first held-out pair
in manifest order, four complete pages. Controls use that first held-out pair,
two pages per intervention. All eight native calls use 50 steps, guidance one,
the unchanged generic condition and 512×256 output. Use separate inference
processes with no training manifest or reader loaded. Require the erased-question
pair's actual output tensors and PNG pixels to match exactly; inspect all eight
pages. Neither exact execution nor four normal answers establishes language
learning or a full-cohort accuracy estimate.

Separately verify the frozen native encoder/decoder on those full source pages,
preserving its posterior mode, normalization and patch geometry. Retain complete
reconstructions for visual/OCR comparison with the originals. Source-image
checksums and ink boxes alone do not prove information survived the encoder.

A two-update gradient/memory preflight uses only the first training question
pair and the unchanged final-image objective, loading targets only after
generation. These are real optimizer updates, subsequently discarded. Require
native-forward equality before the first update, finite nonzero adapter and
early/middle/late trajectory gradients, no frozen parameter gradients, changed
adapters and peak allocated memory below 22 GiB. Keep the shared GPU0 lock and
RAM/swap policy. Do not infer this full-page fit from the shorter copying run.
Do not promote these two-update weights or claim they establish semantics.

Before a full language-learning run, register its complete schedule and objective,
calibrate any training reader on the actual QA answer images, and evaluate the
complete fixed normal/control cohort. Retain earlier writing requirements;
a successful execution preflight cannot substitute for reliable image writing.

## Interpretation boundary

Independent CPU reference OCR on the first 32 training and all 32 held-out
canonical answers recognizes 23/32 and 26/32 exactly. This is evaluator
calibration, not model performance. Keep its complete report and actual
misses. Never choose a transcript from the expected answer or silently repair
generated pixels. Raw review must distinguish readable correct answers,
wrong answers and writing failures. A later semantic evaluator should retain
all source-accepted answers and report both individual questions and complete
same-passage pairs, with paired question/passage controls and full costs.

This public development cohort has informed previous experiments and is not a
pristine project-wide blind test. Historical writing, arbitrary glyph generation,
general instruction following, a lightweight foundation and efficiency at matched
language quality remain separate requirements of the active ILM goal.
