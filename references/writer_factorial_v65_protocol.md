# Writer failure diagnosis: fixed visual information and objective controls

Registered on 5 November 2026, after the completed V64 failure and before
these outputs. Useful independent language remains the goal; this experiment
diagnoses its failing visual output path.

## Question and scope

Does the current writer recover a finite set of writing images when supplied
with inherited visual features? Does changing the conditional information or
the learning objective change that answer? This is a training-only diagnostic
of the actual output architecture. Memorization here is deliberate and cannot
qualify language, held-out generalization or deployable answer generation.

Use the unchanged `ContinuousBlockWriter` from the complete grounded actor:
19,374,096 parameters, six 384-wide layers, physical 32-by-64 pixel outputs
and two 4,096-dimensional conditioning vectors. Do not edit the V64 actor,
loader, trainer or protocol. The already qualified V34 image codec supplies
a separately measured reconstruction reference, without new codec training.
No tokenizer, character ID, OCR, glyph inventory or external service is used.

## Fixed data and conditions

Select 32 CMRC and 32 Dolly training examples uniformly without replacement,
CPU seed 20266510. Interleave sources and select one uniform complete answer
block per example. Exclude the two structural termination blocks. Keep actual
white answer blocks. Use the V64 training cache and its pinned report SHA
`0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e`.
Do not open development or sealed data. Save the complete 64-case manifest,
source hashes, signed pixels and exact reference PNGs before fitting.

Two fixed conditions:

1. **Inherited:** the same 27-layer FP32 encoder, original 32-by-64 image
   resolution, two output vectors and no contextual body computation. Extract
   once from the authenticated source actor, then release the inherited body.
2. **Pixels:** the same image split into left/right physical 32-by-32 tiles,
   each flattened and concatenated as `[pixels, pixels, +1 anchors, -1 anchors]`.
   Each segment has 1,024 entries, giving two 4,096-vectors. This fixed linear
   input preserves every source pixel; the anchors preserve its distinguishability
   under per-vector LayerNorm. It is an oracle control with the target image
   supplied, never a way to generate an unknown answer during deployment.

Both conditions pass through the unchanged writer's normalization/projection.
Record centered feature variation/effective rank descriptively, without
using these measurements to select an arm, schedule or transformation.

## Four fixed learning arms

Run sequentially: inherited/flow, inherited/direct, pixels/flow, pixels/direct.
Every arm starts from the same freshly initialized writer state, model seed
20266520. Each receives the same 2,000 batches of 16 uniform case indices,
sampled with replacement, CPU seed 20266521. These are not continuations of
V64's final writer. Batch exposure and initial weights are identical by hash.

- **Flow:** U~N(0,I), t~Uniform(0,1), Yt=(1-t)U+tY, MSE target Y-U.
- **Direct:** input image and time are zero, MSE target Y. The same writer
  parameters produce the complete image deterministically. Draw and discard
  the same noise/time tensors so random-stream advancement is matched.

Use AdamW 2e-4, betas (0.9,0.95), weight decay .01, clip norm 1, 100-update
linear warmup then cosine decay to 2e-5, FP32 parameters/moments, BF16 writer
contractions, deterministic math attention and EMA .99. Noise/time CUDA seed
20266522 resets per arm. Budget is 2,000 updates per arm, with no early quality
selection. Save current/previous checkpoints, full RNG/optimizer/EMA and batch
journals every 250 updates, at update two and on handled interruption.
At update three, restore update two and require exact replay of parameters,
EMA, optimizer, random state, indices and loss. Count one extra replay per arm.

## Fixed measurements

After each final EMA is fixed, reconstruct all 64 fitting cases under correct
conditions and a two-case cyclic permutation. Interleaving makes this shuffle
stay within source language; it avoids testing only Chinese/English distinction.
Flow uses 16 explicit-midpoint steps with per-case seed 20266530+i; its two
conditions share initial Gaussian noise. Direct uses zero input/time once.
Threshold at zero. Save all 512 generated PNGs across arms/routes, all scores,
exact-block and blank fractions, and first-eight-case galleries without repair.

The diagnostic fitting criterion is foreground F1 >=.95 and correct-minus-
shuffled foreground gain >=.50. It is not the V64 development gate, a language
gate or evidence of generalization. Report each arm and the full factorial
comparison, without selecting only the best outputs. No significance or seed-
robustness claim is made from this finite fitting experiment.

On exactly these training images, load the existing V34 EMA after verifying
checkpoint SHA `a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33`.
Use its unchanged continuous encoder/decoder on the physical tiles, with ink
polarity converted explicitly. Report reconstruction F1/exact/blank only;
its earlier codec qualification does not transfer language capability.

## Interpretation and execution

Direct success with flow failure localizes a problem in the tested flow
training/sampling route; it does not rule out flow methods generally. Pixel
success with inherited-feature failure motivates an interface or optimization
repair; it does not prove irreversible encoder information loss. Inherited
success on this finite set identifies a difference from V64's broad-corpus
training, not adequate language capacity. Any next repair needs a new fixed
protocol before observing its output.

Use one owned GPU-0 job, no GPU-1 service. Check host RAM >=24 GiB available,
swap <75%, GPU 0 idle at startup, allocated CUDA memory <22 GiB and elapsed
time <1 hour. Save on signals/resource interruption; preserve failed attempts.
Record inherited extraction memory separately from each writer arm after
resetting CUDA peak counters. No full-model language-memory advantage follows
from dropping unused inherited modules during this diagnostic.

Entry point: `scripts/diagnose_grounded_writer_v65.py`. Commit/push the frozen
protocol and implementation before execution. Keep the general ILM goal and
the verified V63/V64 Nature manuscript intact while this diagnosis runs.
