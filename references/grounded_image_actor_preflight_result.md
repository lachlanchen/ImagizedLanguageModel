# Complete grounded image actor: measured execution and memory

5 November 2026. The complete real-weight actor passes all **19 preflight
checks** on physical GPU 0. This establishes execution, backward and
resumability within one RTX 4090 D. Readable writing and language answers
remain unproved.

| Measurement | Result |
| --- | ---: |
| Complete actor parameters | 7,548,043,521 |
| Inherited frozen parameters | 7,522,463,984 |
| Trainable adapters, queries, writer and stop head | 25,579,537 |
| Image writer parameters | 19,374,096 |
| Vision / contextual layers | 27 / 36 |
| Peak allocated / reserved memory | 17.181 / 18.053 GiB |
| Full run, including source hashing/loading/geometry | 87.585 s |
| Initialization and setup portion | 77.549 s |
| Native image-packing maximum error | 5.96e-8 |
| Cached/full-prefix absolute / relative maximum error | 0.0003910 / 1.21e-5 |
| Registered relative cache tolerance | 0.02 |
| Same-shape future-pixel / separate-image interventions | Exactly zero / zero |

The geometry scan covers all 12,976 complete training examples. The largest
contextual row has 918 positions: a 480-by-1,024 prompt and 109 complete target
blocks. The largest prompt is 512 by 1,024, with 30 target blocks. Both receive
complete contextual forward/backward, with no truncation of prompt or answer
history. Sixteen uniformly selected blocks receive writer loss. Adam moments
are allocated before these resource cases execute.

All trainable parameters receive finite gradients; inherited parameters receive
none. Nonzero gradients reach first/last contextual adapters, continuous
queries and the writing head. Restoring a saved update against the same frozen
body reproduces every trainable weight, optimizer state, CPU/CUDA/local noise
RNG state, selected block and update metric exactly. Independent CPU inspection
of the serialized checkpoint confirms 294 finite tensors containing 25,579,537
trainable parameters and optimizer steps equal to one. No frozen body or
language vocabulary matrix is stored in that checkpoint.

## Preserved failures and repair

Attempt 01 fits at 15.195 GiB but fails the cache check at relative error
0.18524. An actual-weight diagnostic keeps features and coordinates identical
and still finds 0.08514; FP32 residuals alone leave 0.07507. Attempt 02 uses
FP32 linear projections but leaves attention under ambient BF16 autocast,
and still fails at 0.09815.

Attempt 03 disables autocast over the entire contextual call, including
attention products and checkpoint recomputation. Contextual parameters retain
BF16 storage with layer-local FP32 casts; the smaller vision body uses
source-derived FP32 weights/computation. Writer contractions may use BF16.
An attention-entry hook checks the precision boundary. All original thresholds
remain unchanged. The imported quantized/BF16 precision loss is not reversed.
The receipt binds both failed reports and the diagnostic.

## Scope

Each 32-by-64 answer block is independently image-encoded before causal context
processing. Two continuous queries request its 2,048 output pixels, generated
jointly by a conditional flow. A separate structural classifier learns stopping.
The contextual vocabulary module is removed. No tokenizer, Unicode IDs, glyph
bank, OCR or external model service is called.

Two optimizer updates and replay are execution checks. The fixed two-block
output is still noise and reaches its execution cap; it is preserved without
repair and proves no readability or semantics. Native contextual equivalence
to another engine and a fresh standalone offline deployment bundle remain
unverified. Only training data is opened; no development or sealed examples
are used. Six CPU image-boundary tests also pass against the native processor.

The next experiment must teach coherent writing and ordinary Chinese/English
instruction answers, with autonomous-history and question controls.
Reconstruction alone cannot close the language goal. No efficiency or general
model parity is claimed: total size is 7.548B, even though only 25.58M parameters
are trainable. External Qwen3-VL pretraining remains attributed.

See the [plan](grounded_image_actor_preflight_plan.md),
[receipt](evidence/grounded_image_actor_preflight.json) and
[implementation guide](../docs/grounded-image-language.md). Full local evidence
is in `artifacts/grounded_image_actor_preflight_20261105/`, with
`attempt03/report.json` the accepted execution report. The owned jobs and tmux
sessions exit normally and release GPU 0. No GUI was started.
