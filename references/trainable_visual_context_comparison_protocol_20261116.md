# Complete-corpus comparison of frozen and learned visual context

The [real update/reset preflight](trainable_visual_context_preflight_result_20261116.md)
passes on complete training records with a peak allocation of 11.252 GiB. Its
six trials are reset and are not language evidence. This protocol registers the
subsequent full training comparison. The selected variable is whether the whole
1,409,410,048-parameter inherited contextual body learns. It tests the compact
model's frozen-context limitation without claiming that unfreezing is sufficient.
Earlier smaller trainable bodies and the full-rank frozen body failed language.

## Model, sources and matched arms

Use the exact preflight model and six equally weighted supervised objectives,
fresh initial image encoder and 1,024-wide full-rank flow writer, frozen codec,
training seed 202611152, FP32 parameters and BF16 forward autocast. Both arms
use the same host-moment ordinary AdamW. Keep learning rate 2e-4 for the image
heads and training-only projection, 2e-5 for the treatment's complete context,
weight decay .01, betas (.9,.999), epsilon 1e-8 and global gradient clip 1.
The control freezes context. Both retain gradients from the writer through the
body to the image reader. Source vectors and saved teacher states supervise
training only. No rejected judge, development output as fitting data, reserved
DRCD record, runtime teacher, lexical output head, renderer or answer lookup.

Use every original training record: **13,785 sequences, 2,576,239 target glyph
positions and 2,380,620 saved teacher states per arm**, with all 108 shards.
Bind every source/pixel/alignment shard and all cached states, code, codec,
package versions and evaluation images before fitting. The 7,255,223 total
source cells have already passed exact renderer fidelity. Keep physical aliases
and uncertain readings; do not replace this supervision with pointwise Unicode
classification. Prior inherited text pretraining remains a disclosed source of
language capability. This is visual adaptation, not proof of learning language
from images alone or 1,000-fold efficiency.

## Fixed complete schedule and storage

One epoch per arm. Use the original shard shuffle and within-shard batch shuffle
with seed 202611152, padded native-unit budget 1,024 and complete records. There
are 8,172 updates per arm. Log every actual batch's source indices, all six losses
and gradient norm. Save deployment snapshots at initialization and after shards
27, 54, 81 and 108; register their exact update numbers from the full schedule.
Save the latest full training state every nine completed shards, including host
optimizer moments, teacher projection, counters and RNG states. No automatic
restart or unchanged extra epoch follows an error.

Learned-context deployments must contain **every FP32 context tensor** plus the
learned image encoder/writer. Frozen-context deployments reuse the authenticated
original context instead of duplicating it. Each actual initialized full model
undergoes a byte-exact snapshot/restore check before fitting. Preserve the five
fixed snapshots. The CPU completion audit reconstructs the entire expected batch
order, visits every training index exactly once, reconciles source/state counts,
checks all final optimizer counters and moment sizes, and compares complete final
deployment tensors with the latest full training state. Only audited complete
fits proceed to their prescribed generation. Do not select an intermediate as
the final model. No independent full optimizer-trajectory recomputation is
claimed by this final-state/source audit.

The sole serial owner runs treatment fit/audit/generation, then control fit/audit/
generation. It verifies identical initial complete model/objective hashes across
arms. Outputs cannot change the committed control recipe. A failed stage ends
the pipeline and preserves partial evidence. No second heavy project worker.

## Complete autonomous evaluation

Keep all 120 existing reference-free fixed-cell inputs: 32 normal QA questions
in sixteen passage pairs, both question- and passage-erased versions, twelve
normal book prefixes and twelve erased book prefixes. The whole held-out works
are 欒城集 and 西遊記. Initial and final snapshots generate all 120 with cap 128;
the three intermediate snapshots generate all 44 normal inputs. The final model
also generates all 24 normal/erased book streams with cap 640. This totals
**396 streams per arm, 792 overall**, including every failed, blank or capped
response. Use eight flow steps, seed 202611153 and BF16 forward autocast in both
arms. The original compact comparison used different numerical placement;
retain it as prior evidence, and use the fresh matched control for this effect.

Generation consumes only input pixels and learned/authenticated model weights.
Every generated span is decoded to actual binary pixels and reread into causal
history before further generation. A zero-length span is a natural stop; hitting
the cap is censored. Preserve raw tiles, display PNG, every feedback hash, stops,
caps, seeds, timings and complete checkpoint identity. Do not recode answers with
a font, replace emitted pixels or feed reference history during evaluation.
After each arm's generation, replay the fixed final `book_01_normal` input in
a traced fresh process, empty working directory and empty Hugging Face cache.
Require identical pixels, stopping and feedback hashes, no IP connection,
tokenizer/font/corpus access or undeclared artifact reads. These two replay
calls are additional execution checks, outside the 792 language streams.

Review every full stream under opaque identifiers before reference scoring.
Normal books include their actual visible prefix for coherence review. Record
literal readings, uncertainty and complete-continuation coherence; freeze all
792 readings and image hashes before opening new reference answers. Then apply
the preserved whitespace-normalized strict complete-answer rule, which requires
readability and natural stopping. Count sixteen both-correct question pairs,
readability, both erasure arms, normal-minus-best-erased dependence and complete
long-book coherence separately. The exposed development cohort and model review
do not establish untouched generalization or independent human adjudication.

Final treatment development support requires at least 12/32 strict exact,
4/16 both-correct pairs, 24/32 readable and 4/12 coherent naturally stopped long
book continuations, each QA erasure at most two exact and dependence gap at least
eight. Report all matched control differences, even if the control also passes;
actor quality and intervention benefit are distinct. A pass permits a separately
bound untouched confirmation comparison. This fixed-cell stage does not measure
the full goal's proportional presentation, earlier-success retention under those
presentations, historical/unencoded forms or etymology. None is waived.

On failure preserve all outputs, checkpoints, scores and uncertain readings.
Do not relax thresholds, tune a reward, repair transcriptions or select a better
intermediate. Choose a further bounded change only from the observed failure.
The goal stays active until the full acceptance criteria are demonstrated.

## Resource ceilings and reporting

Pin physical GPU0, PCI bus ordering, CUBLAS_WORKSPACE_CONFIG `:4096:8` and two
CPU threads. Use the existing project GPU mutex. Require 48 GiB available host
RAM and 20 GiB free GPU at every heavy admission, retain 24 GiB host RAM and
at most 22 GiB allocated GPU memory. Each arm has an 86,400-second fit ceiling
with at most 1,800 seconds of pressure waiting. Keep at least 48 GiB free disk;
the current 335-GiB free space admits full FP32 snapshots and rotating latest
states without copying shared model archives. Each inference snapshot has a
7,200-second ceiling and the same host/GPU limits. Guard every contextual forward
during generation. Preserve foreign GPU1 services; no process cleanup outside
verified project ownership.

Report whole deployed and trainable parameters, inherited pretraining and saved
teacher preparation separately, actual record/position/state exposures, all fit/
generation wall time and waiting, peak VRAM, host minimum and checkpoint sizes.
A lower training loss is not a language score. A single-GPU fit is not a
matched-quality compute advantage over a local Qwen baseline. The Nature-style
manuscript and native figures must use completed measurements and original
outputs, with failed baselines preserved.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python -m unittest tests.test_offloaded_adamw tests.test_trainable_context_checkpoint -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/register_visual_context_comparison.py
# Commit and push every bound source, then launch the sole serial owner once.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. OPENBLAS_NUM_THREADS=2 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/run_visual_context_comparison.py
```
