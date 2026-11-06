# Retained writing conditions survive the tested BF16 interface

The completed V81 writer still fails, but its saved conditions do not show
hard quantization aliasing at the first writing-head projection. A CPU audit
uses every pre-block full-history state on all 32 training questions, in both
completed arms: 214 states across 64 traces. This is an auxiliary diagnostic,
not a new language result or a change to the running V82 comparison.

| Completed V81 arm | Writing conditions | Unique FP32 states | Unique BF16-rounded inputs | Unique CPU BF16 projections | Median projection relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Continued control | 107 | 107 | 107 | 107 | 0.191% |
| Adapted readout | 107 | 107 | 107 | 107 | 0.182% |

All six tested representations retain 107 distinct rows in each arm,
including FP32 projection, input-only rounding, and rounded operands with
FP32 accumulation. No exact collision group combines differing target
images. The median input-rounding error is 0.157% for control and 0.163%
for adaptation. Maximum CPU BF16 projection error relative to FP32 is
0.265% and 0.221% respectively. These are feature-vector errors, not pixel
fidelity or semantic accuracy.

Each arm has 42 matched block positions between the same-passage question
pairs; all have differing target blocks. Their projected distance ratios
after CPU BF16 computation range from 0.999402 to 1.000421 in control and
0.999644 to 1.000509 after adaptation. Thus the tested rounding operation
does not erase those pairwise distinctions. Distinct states and preserved
distances do not establish that the downstream predictor uses them correctly.

Independent NumPy verification reconstructs every original target-block hash
and retained condition, checks exact BF16 input-rounding bits, verifies all
collision groups and all matching pairs, and recomputes the error statistics.
It also checks Gram-based stable/entropy spectral summaries and independently
reconstructs the FP32 projection from the authenticated trained weights.
The latter agrees within relative L2 error 2.49e-7. Numerical rank is retained
as a producer measurement, not independently estimated by that verifier.

CPU BF16 and the rounded-operand FP32 emulation differ in 36 of 164,352
projected values for control and 22 for adaptation. Neither implementation
is a replay of the GPU reduction kernel. The audit covers the condition
interface; it does not bound errors in every later BF16 residual layer or
replay a full flow trajectory. A lossy representation need not have an exact
collision, so subtler information loss remains possible.

The measurements weaken the proposed hard-aliasing explanation for these
retained training conditions. They do not identify the cause of the corrupted
answers or justify treating an FP32 projection as an established repair.
The matched decoder-sensitive loss experiment continues unchanged.

The [receipt](evidence/writer_condition_precision_v82_result.json) retains
every row, source and checkpoint digest, pair distance, spectrum and
verification scope. The original arrays and projection weights remain under
the authenticated artifact path. No complete contextual model, optimizer,+sampler, new answer or additional GPU job is used in this audit.
