# Qualified writing does not settle current-cohort language

The frozen balanced writer reproduces all 64 selected training glyphs exactly:
32 first and 32 later positions. It passes both descriptive per-pool physical
gates on these saved states. Its complete current-cohort language evaluation
is registered separately and remains unscored while generation is running.
No model fitting is performed by this diagnosis.

| 32 positions per group | Balanced mean F1 | Balanced exact tiles | Parent mean F1 | Original flow mean F1 |
|---|---:|---:|---:|---:|
| Training first | 1.000000 | 32 | 0.872790 | 0.984237 |
| Training later | 1.000000 | 32 | 0.963619 | 0.954535 |
| Fixed-cell development first | 0.479076 | 0 | 0.492173 | 0.447771 |
| Proportional development first | 0.658382 | 4 | 0.632918 | 0.661558 |

The development images include changed components and damaged forms. Canonical
pixel mismatch can also reflect different answer prefixes or learned stroke
variation; it is not a semantic-accuracy label. The first-image comparison
never supplies correct history to autonomous generation. Supplied-codec
reconstruction remains 31/32 first and 32/32 later training tiles exact.
Different conditional motor coordinates can cross the binary threshold
correctly even when direct supplied reconstruction differs by a pixel.

No sampled nonterminal state crosses the balanced stop threshold of 0.5.
The previously observed later false stop changes from 0.99542 to 0.00679.
This diagnosis writes one glyph regardless of stopping and cannot establish
proper ending during generated history. The trained readout and stopping
predictor are evaluated together; the effect is not isolated to the writer.

Registration authenticates the fixed balanced checkpoint and its earlier
complete fit/qualification/audit. Direct CPU comparison finds 26 image-codec
and normalization tensors unchanged from the parent; only twelve readout/stop
tensors differ, and all inherited contextual adapter tensors are bit-exact.
All model parameters stay frozen during this study. The previously qualified
checkpoint is fixed before this diagnostic and is not selected from its scores.

Neural diagnosis and scoring take 1.301 seconds at 0.217 GiB allocated GPU
memory, with no pressure waiting. Terminal wall time including board export
is 1.677 seconds. The full inherited reader is not loaded. This excludes its
7.522B visual/contextual parameters, earlier fitting, preparation and CPU
validation/review costs; it is not an efficiency comparison with an LLM.

An independent CPU pass recounts all 128 pixel/motor scores and verifies all
512 displayed tile arrays. All eight labelled boards are inspected with
prior cohort familiarity, without independent human adjudication. A first
CPU recount stops at a stale archive key from the earlier paired-flow auditor;
the corrected reader recounts the same saved files without new model sampling.
The failed log is retained. Raw runtime process/session arguments are kept
private; the public evidence retains resource counts and identity bindings.

These results demonstrate faithful conditional writing on the sampled
training states. They do not demonstrate useful complete language or explain
the development gap by a single cause. Finish all 216 unchanged autonomous
streams, freeze whole readings before reference scoring, and compare retained
exact answers, pairs, readability, books and erasures. Do not begin another
writer fit or substitute this physical result for that language test.

See the [prospective decision](balanced_writer_matched_decision.md),
[registration](evidence/balanced_writer_matched_registration_20261111.json),
[complete diagnostic record](evidence/balanced_writer_matched_diagnosis_learning_20261111.json)
and [pixel archive](evidence/balanced_writer_matched_diagnosis_20261111/pixels.npz).
All boards remain available:
[1](evidence/balanced_writer_matched_diagnosis_20261111/board_01.png),
[2](evidence/balanced_writer_matched_diagnosis_20261111/board_02.png),
[3](evidence/balanced_writer_matched_diagnosis_20261111/board_03.png),
[4](evidence/balanced_writer_matched_diagnosis_20261111/board_04.png),
[5](evidence/balanced_writer_matched_diagnosis_20261111/board_05.png),
[6](evidence/balanced_writer_matched_diagnosis_20261111/board_06.png),
[7](evidence/balanced_writer_matched_diagnosis_20261111/board_07.png),
[8](evidence/balanced_writer_matched_diagnosis_20261111/board_08.png).
