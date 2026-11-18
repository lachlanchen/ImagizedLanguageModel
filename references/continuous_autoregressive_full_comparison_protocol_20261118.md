# Complete-corpus native continuous-density comparison

The complete-record preflight and independent 2,208-image recount pass. This
protocol admits a new comparison using the qualified V34 continuous decoder;
it admits neither the rejected binary codec nor the failed correction motor.
No earlier completed fit, image review, source audit or preflight is repeated.

Compare K=1 and K=8 in the implemented native causal model. Preserve all common
initial tensors, model seed 202611198, source order seed 202611199, FP32 strict
deterministic execution and source pixels. The candidate has 29,799 additional
output parameters. Both include the same frozen 7,423,361-parameter trained
codec and have no inherited language weights. Neither arm receives teacher
states, a tokenizer, source embeddings, glyph IDs or a font during execution.
Offline corpus rendering and teacher-written source material remain preparation
costs. This is a new native language-density comparison, not a controlled
replication of the inherited 1.465B actor.

## Retained fitting schedule

Use all 108 training pixel shards, 13,785 complete records in one fixed shuffled
permutation shared by both arms. Preload only the packed pixel arrays/offsets
under the 8-GiB RSS ceiling. Keep all source images. The first image anchors
book pretraining; QA uses the original prompt boundary. This yields 2,571,477
next-image targets and 2,585,262 stop decisions per arm. The earlier 2,576,239
source target-position count includes the 4,762 first book images, which this
causal objective uses as inputs rather than next-image targets. Do not call
these two exposure counts equal. Maximum record length is 1,101, below the
fixed 2,048-image context; no training truncation is allowed.

One complete record produces one AdamW update, 13,785 retained updates per arm.
Use betas (.9,.95), weight decay .01, global norm clip 1, 250-update warmup to
2e-4 and cosine decay to 2e-5. Optimize mean code NLL per coordinate plus mean
stop BCE, without new losses, weights, rewards or augmentation. Record every
actual record index, target/stop count, learning rate, both losses and gradient
norm. Save whole-model deployment snapshots at updates 0, 3,447, 6,893, 10,339
and 13,785; no quality-based checkpoint selection. Retain optimizer counters,
source order, full ledger and original failed states. No extra unchanged epochs.

## Fixed autonomous evaluation and acceptance

Bind the existing reference-free 120-input compact evaluation manifest and
every input PNG before training. It contains 32 normal paired QA questions,
32 question-erased and 32 passage-erased inputs, plus twelve normal and twelve
prefix-erased book inputs. Initial/final snapshots generate all 120 at cap 128;
the three intermediate snapshots generate the 44 normal inputs at cap 128.
Each final snapshot also generates all 24 books at cap 640. This gives 396
streams per arm, 792 overall, with no target lengths or reference images at
inference. Use fixed greedy component means, the original learned stop threshold
zero, actual emitted-pixel feedback and all natural stops/caps. Review all
outputs, including blanks and unreadable writing; freeze anonymous readings
before new reference scoring. Do not use source images, font rerendering, OCR
or reference text to repair any output or uncertain reading.

An independent CPU completion audit must first verify the entire expected
training order, 27,570 updates, 5,142,954 target exposures, all ten snapshots
and latest optimizer counters. Generation requires the exact trained snapshot
and committed execution sources. CPU pixel recount and stopping are separate
from actual frozen language review. Compare final complete questions, paired
correctness, readability, natural stopping and coherent complete books. Apply
the original 12/32, 4/16, 24/32 and 4/12 development requirements, erased-control
limits and dependence gap; intermediate outputs are trajectory evidence only.

This fixed-cell cohort cannot by itself establish the goal's proportional
presentation or retention of its eight proportional/two fixed earlier successes.
Those remain required in a separate prospective extension if useful language
appears. The untouched DRCD confirmation pool remains unrendered, unscored and
excluded from every training/calibration path. Historical/unencoded etymology,
different fonts, cold independent execution and matched-quality whole-model
cost also remain required. This fit cannot complete the full goal by itself.

If the final model fails, preserve every result and reject this unchanged
recipe as sufficient. Do not infer success from NLL, natural stopping, a single
legible word or a better intermediate. If useful complete language passes this
development comparison, require the remaining retained-success/proportional,
independence and untouched-source confirmation before a broader claim. No
1,000-fold gain follows from model size or this small training exposure count.

## Resources and ownership

One serial fit owner trains K=1 then K=8 under the GPU0 mutex. Require 48 GiB
available host RAM and 20 GiB free GPU at entry, retain 24 GiB host RAM and
48 GiB disk, maximum 22 GiB GPU allocation and 8 GiB process RSS, two CPU threads.
The combined fitting deadline is 7,200 seconds including at most 300 seconds
pressure wait. Outer termination is 7,260 seconds plus thirty seconds grace.
Observe the actual process identity; never restart from an observation timeout.
After fit exit, audit on CPU, then admit one separate serial evaluation owner
under the same reserves with a separately bounded 7,200-second generation phase.
Any timeout or resource failure is preserved, with no automatic retry. Whole
cost reports include both arms, audit/generation, preparation and prior codec
training separately. The session goal remains active.
