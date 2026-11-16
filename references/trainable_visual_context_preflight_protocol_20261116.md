# Full-context visual learning: execution preflight

The compact full-rank model completed its original full-corpus fit without a
strict complete answer on the 32-question development cohort. Its 1,409,410,048
contextual parameters were frozen. Widening the image writer removed a measured
output-rank restriction, but did not establish language. The selected comparison
lets this entire contextual body adapt to visible writing. Earlier V35 trained
a smaller body and also failed; trainable context is neither a novel principle
nor a sufficient explanation for the observed failures.

This preflight checks actual complete-record optimization and memory support.
It does not fit a successor model or qualify language. A full matched-corpus
fit, learned-context checkpoint loader and complete autonomous evaluation need
their own prospective registration after this execution check.

## One change and matched control

Both arms start from the authenticated Qwen3 1.7B contextual archive with its
lexical embedding removed, the same frozen continuous image codec, a fresh image
encoder and the same 1,024-wide full-rank image writer. In both arms contextual
parameters are FP32 and forward arithmetic uses BF16 autocast, with gradient
checkpointing. Both use ordinary FP32 AdamW with CPU moment storage and bounded
GPU updates. The treatment makes every contextual parameter trainable. The
control freezes them. It retains gradients through the frozen body to the image
encoder. No tokenizer, lexical output head, renderer, OCR or live teacher is
used for model input or answer production.

Preserve the six equally weighted original training objectives: source-feedback
vector alignment, update counts, span boundaries, next-span length, next-span
continuous image flow and saved teacher-state alignment. Native source vectors
and teacher states are offline training supervision only. Use learning rates
2e-4 for image heads and the teacher projection, 2e-5 for the treatment context,
weight decay .01, betas (.9, .999), epsilon 1e-8 and global gradient clip 1.
Offloading changes storage and transfer, not the AdamW learning rule. Count the
whole inherited model and preparation cost; it is not a from-scratch foundation.

## Complete authentic examples and actual updates

Scan all 13,785 original training records and bind their report and source hashes.
Select the first QA record, first complete 513-glyph book and record with the
greatest native-unit length, breaking ties by corpus order. Bind their complete
pixel/source/alignment shards and saved teacher-state files. No substring,
development answer, reference image, rejected image reward or reserved DRCD
record enters these losses.

Run the frozen control first, then the trainable-context arm, on all three
complete cases. Restore identical full initial model/objective state and CPU/
CUDA random state for each trial. Initial six losses must agree across arms to
absolute 1e-6. Require finite loss/backward; all context tensors must receive
finite nonzero gradients in treatment and none in control. Apply a real update
to every active parameter. Require every context tensor to change only in the
treatment and one actual Adam counter and host-resident moment pair for every
active tensor. The CPU unit check independently compares multiple AdamW steps
and resume behavior with PyTorch. During this real GPU action additionally
compare up to seventeen evenly spaced entries of every active tensor, using
its actual clipped gradients, against an independent GPU PyTorch AdamW step
(rtol 1e-6, atol 1e-7). This samples update arithmetic; it is not an independent
full-model gradient derivation.

Measure the same complete supplied-history losses after each update under the
same random state. They are diagnostic; neither a one-step decrease nor a low
training loss qualifies autonomous language. Reset every model parameter,
persistent buffer and objective parameter exactly after every trial, verifying
full-state byte hashes. Keep the codec frozen and lexical head absent. Retain
all six update/reset receipts, including failures. No trained checkpoint from
this action becomes a deployed model.

## Resources and decision

One serial worker owns the existing GPU0 mutex. Require 48 GiB host RAM and
20 GiB free physical GPU0 at admission; retain 24 GiB available host RAM, at
most 22 GiB allocated GPU memory and 48 GiB free disk. Cap the action at 1,800
seconds with at most 300 seconds of pressure waiting. Use two CPU threads,
PCI bus device ordering, CUDA_VISIBLE_DEVICES=0 and CUBLAS_WORKSPACE_CONFIG
`:4096:8`. Bind package versions. Do not stop foreign services or start another
model worker. Keep live process identity in the private handoff.

Any failed condition stops this exact preflight, preserving partial files and
preventing automatic retry or a full fit. A pass establishes execution support
only. Register all original training sources, both complete matched schedules,
checkpoint restoration, complete emitted-pixel evaluation, stopping, erasures,
resource accounting and an action for either result before a successor fit.
The language goal and all original negative results remain active. No 1,000-fold
compute or performance improvement is established or presumed.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python -m unittest tests.test_offloaded_adamw -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/preflight_trainable_visual_context.py register
# Commit and push the immutable registration before running once.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. OPENBLAS_NUM_THREADS=2 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/preflight_trainable_visual_context.py run
```
