# Diagnose the actual third-update replay failure

The [loaded motor trial](evidence/residual_motor_replay_failure_20261118.json)
passes the complete-source gradient/update/reset path and generates all 68,948
initial images equal to the unchanged base. It retains two updates, then fails
its exact third-update model/optimizer replay. No final trained image result
exists. The failure handler saves the unaccepted third-update attempt, not a
qualified two-update checkpoint. Preserve all files and readings. The original
combined assertion does not retain which component differs; do not invent it.

The source sets deterministic cuDNN and cuBLAS options but does not enable
PyTorch's global deterministic-algorithm requirement. The writer also uses CUDA
embedding-gradient reductions and scaled-dot-product attention. These are
candidate explanations, not an established cause. The installed torch version
is 2.11.0. [PyTorch's deterministic-algorithm documentation](https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html)
distinguishes this global setting from cuDNN convolution controls. It is not a
claim of reproducibility across versions or devices.

Run one prospectively bound numerical diagnostic with two modes: the original
global flag false and the sole intervention true with warn_only false. Keep the
actual saved initial refinement tensors, frozen source cache, exact source order,
first three batches, FP32, existing cuDNN/cuBLAS flags, loss, optimizer and learning
rates. Do not alter attention code, normalization, precision, physical targets or
quality criteria. No classifier, new data, image generation or retained fit.

For each mode, reconstruct its first two updates from the common initial state.
Save that mode's model and optimizer, then repeat the actual third update four
times, restoring the exact same saved model and optimizer before each. Record
loss, gradient norm, CPU/CUDA RNG digests, every parameter-gradient digest and
every model/optimizer digest. Export per-tensor discrepancies when they occur.
This reconstructs a numerical test; it does not claim recovery of an unsaved
original optimizer state. The two modes' post-two states may differ numerically.

All twelve trial updates / 1,536 image exposures are diagnostic and discarded.
Reset both model instances exactly to the common initial tensors and discard
optimizers. Keep the source cache, failed checkpoints and frozen codec unchanged.
Use one GPU0 mutex owner, 40-GiB host entry, 24-GiB live reserve, 20-GiB free GPU,
48-GiB disk, two threads, 8-GiB GPU/RSS caps and 240 seconds including at most
60 seconds pressure wait. External timeout 300 seconds plus 30 seconds grace.
No automatic retry or full language fit.

If the original mode exhibits unequal gradients/updates and the global mode is
exact across every repeat, qualify that execution setting for a separately
registered fresh 6,000-update fit. Do not silently resume the failed model or
discard its outputs. If the original mode does not reproduce its failure, report
that uncertainty; if the deterministic mode fails or throws, stop before a fit
and use the recorded discrepancy to select the next numerical intervention.
Strict replay remains strict. No semantic or physical-quality improvement follows
from making arithmetic reproducible.
