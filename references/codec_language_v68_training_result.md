# V68: completed causal image-language instruction training

Checked 5 November 2026. The registered instruction run has completed all
10,000 optimizer updates on physical GPU 0. This is completed training and
execution evidence. The separate, source-frozen 384-generation
[autonomous audit](codec_language_v68_result.md) is now complete and
independently verified: the actor fails its Chinese answer-quality gates.
The general ILM goal remains open.

## Measured execution

| Quantity | Completed run |
| --- | ---: |
| Total actor parameters | 7,539,239,282 |
| Trainable parameters | 9,351,937 |
| Optimizer updates | 10,000 |
| Recorded example draws | 20,000 |
| Distinct admitted training examples | 12,976 |
| Chinese / English draws | 4,926 / 15,074 |
| Complete first epoch | 12,976 examples, each once |
| Unique draws in the second epoch | 7,024 |
| Journaled sampled block-loss indices checked | 219,066 |
| Peak allocated GPU memory | 16.9653 GiB |
| Recorded training-stage time | 16,773.459 seconds |
| Separate initialization time | 82.717 seconds |

Time is the trainer's measured local execution, including its registered early
optimizer replay and checkpoint work. It excludes external foundation
pretraining, prior codec training, data preparation and the subsequent language
audit. Allocated GPU memory is not total device reservation or an efficiency
comparison against a token model. The inherited foundation remains the main
part of the actor; this run is parameter-efficient adaptation rather than
training a 7.54-billion-parameter foundation from scratch.

The model retains the visual reader and contextual body, with no language
vocabulary module. It learns continuous writing codes from prompt pixels and
causally restricted answer history. The frozen image decoder receives the
predicted codes, and generated binary image blocks are read back through the
visual reader. The [registered protocol](codec_language_learning_v68_protocol.md)
and [actor implementation](../ilm/visual_lm/codec_visual_language_actor.py)
define the unchanged loss and inference boundary.

## Independent checks of the completed records

The [completed-training receipt](evidence/codec_language_v68_training_integrity.json)
checks every recorded update and draw against fresh seeded full-corpus
permutations. All 3,193 Chinese and 9,783 English admitted examples occur
exactly once in the first epoch. The second permutation, cursor 7,024 and
saved CPU data-generator state match exactly. Target block counts and all
sampled block indices agree with the hash-bound accepted manifests.

All journaled metrics are finite and their learning rates match the registered
schedule. All 9,351,937 saved trainable parameter values and 663 optimizer-state
tensors are finite. The final adaptation exactly matches the trainable values
in the update-10,000 production checkpoint. The separately recorded early
production replay passes all ten optimizer, parameter, RNG and sample checks.

This CPU check makes no model forward call or GPU allocation. It verifies
recorded data selection, saved state and schedule consistency; it does not
recompute all training losses or replay 10,000 gradient updates. In the JSON,
`selected_block_losses_verified` counts checked loss **indices**, not
independently recalculated neural losses. Language learning cannot be inferred
from the training loss, source coverage or these integrity checks.

The immutable local run is `artifacts/codec_language_v68_20261105/`:

- Training source commit: `ab43c1ef0ab77b0de2bba622d0372c2c7f69c69a`.
- Training-report SHA-256: `02fe4b868ecb283dd3392acfa63ebd97e731b2462ba616e7633ae6c0c933dc24`.
- Final-adaptation SHA-256: `39d978ae7eefa4fff5a73084e7dd5b262ceb09ecd09b253e2d3627fe03a4b66c`.
- Final-checkpoint SHA-256: `89b67128f1f81e66c6b9544897d1890d2d121086848404821c463324afb4917d`.

## Remaining decision

The original trainer and its queued evaluator have exited. The completed
audit uses the fixed final adaptation and the
[matched-layout control correction](codec_language_v68_layout_control_correction.md).
All normal/control outputs and the original quality gates are retained. The
actor produces 0/128 exact Chinese answers and 32.6713% foreground F1; question
dependence and stopping pass, while answer quality fails. No standalone bundle has been
exported, and the genuinely sealed final split remains unopened.
