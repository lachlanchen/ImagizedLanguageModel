# Decoded endpoint supervision fails the writing gate

The preregistered late-time pixel-loss recipe fails both physical-writing
gates. Its full language trial is therefore not run. This rejects the tested
recipe as a sufficient physical repair, without rejecting all decoder-aware
objectives or continuous language models.

The fit makes exactly 12,000 updates with 24,477,696 trainable parameters.
Initialization, sampled-index digest, final CPU/CUDA random-generator states
and 768,000 exposures from each pool match the original flow control. Decoder
and other bridge weights stay frozen. The only objective change is balanced
decoded BCE at times at least 0.75, with coefficient 1.0. Final EMA, sampler,
seeds, targets and diagnostic positions remain fixed. Six gradient/runtime
checks pass; CPU audit verifies all optimizer parameter steps and finite weights.

| 32 training positions per pool | Control median F1 | Pixel-loss median F1 | Control / pixel-loss count ≥0.95 | Control / pixel-loss exact bitmaps |
|---|---:|---:|---:|---:|
| First | 0.986408 | 0.987034 | 32 / 32 | 4 / 6 |
| Later | 0.974518 | 0.976561 | 23 / 23 | 0 / 1 |

Each pool requires median F1 at least 0.99 and at least 30/32 scores at least
0.95. Both fail. Later mean F1 changes from 0.954535 to 0.953115. More exact
training tiles do not establish consistent physical improvement. The supplied
codec remains 31/32 first and 32/32 later exact. Residual stroke errors and
damaged forms persist in the complete eight boards.

All 64 normal development first states are sampled once with original seeds,
without a contextual-body call. Canonical mean F1 is 0.451322 fixed-cell and
0.658197 proportional, versus 0.447771 and 0.661558 for the control. These are
supplied-state first images, not autonomous answers or semantic accuracies;
native/canonical wording can differ. No new whole-answer reading, exact-QA
score or book result is assigned. Existing complete-answer results remain
those of the original flow trial.

Fitting takes 130.299 seconds at 1.169 GiB peak allocated GPU memory, with no
pressure waiting. The 128-position assessment takes 22.577 seconds at
0.311 GiB. The full reader is never loaded. All 128 saved pixel counts and
normalized motor errors are independently recounted; all 512 displayed tiles
are checked against raw arrays and visually inspected. Control pixels, paired
noise and stopping values are unchanged. This is a labelled model review
without independent human adjudication.

The endpoint estimator is trained on noise/target interpolants. Its small
training BCE does not directly measure the endpoint reached by its own ODE
rollout. That mismatch, conditional transfer and stopping remain unresolved;
none is established as the sole cause. Do not extend this fit, change its
coefficient or select another development sample. A different realized-endpoint
training test needs its own decision after the manuscript update. No usable
language or task-efficiency claim follows.

See the [decision and math](decoder_aware_motor_decision.md),
[registration](evidence/decoder_aware_motor_registration_20261111.json),
[fit audit](evidence/decoder_aware_motor_fit_20261111.json) and
[complete assessment](evidence/decoder_aware_motor_learning_20261111.json).
The [raw archive](evidence/decoder_aware_motor_20261111/pixels.npz) and boards
[1](evidence/decoder_aware_motor_20261111/board_01.png),
[2](evidence/decoder_aware_motor_20261111/board_02.png),
[3](evidence/decoder_aware_motor_20261111/board_03.png),
[4](evidence/decoder_aware_motor_20261111/board_04.png),
[5](evidence/decoder_aware_motor_20261111/board_05.png),
[6](evidence/decoder_aware_motor_20261111/board_06.png),
[7](evidence/decoder_aware_motor_20261111/board_07.png) and
[8](evidence/decoder_aware_motor_20261111/board_08.png)
preserve every position with no pixel repair.
