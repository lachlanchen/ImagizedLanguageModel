# Global deterministic mode qualifies the observed optimizer replay

The [registered actual-source diagnostic](evidence/residual_motor_replay_diagnostic_result_20261118.json)
reproduces unequal optimizer states under the original execution setting.
Across the three comparisons with the first third-update repetition, losses,
gradient norms and model weights are identical, while gradients of depth, row
and column embeddings differ by at most 5.820766091346741e-11. Six optimizer
moment tensors differ. CPU/CUDA random states do not change.

With `torch.use_deterministic_algorithms(True, warn_only=False)`, all four
repetitions have identical losses, norms, every gradient, model tensor and
optimizer state. Both modes start from the actual saved initial weights and
reconstruct the first two source batches before testing the third. This does
not recover the original unsaved optimizer state or identify every possible
operator-level cause. It directly qualifies the global execution setting on
the observed batch without relaxing strict equality.

All twelve trial updates / 1,536 image exposures are discarded, with both
models reset exactly and the frozen codec/source files unchanged. No glyph is
generated. The diagnostic takes 1.674 monitored seconds at .310 GiB peak
allocation and 1.536 GiB process RSS; supervisor time is 10.011 seconds. Those
component costs exclude earlier preparation and any contextual foundation.

This fixes an observed reproducibility problem. The differences are tiny and
do not establish a cause of unreadable language in earlier models. No final
motor quality, semantic capability or efficiency is measured. A fresh full
physical fit may now be separately registered with this one execution change,
the original initial weights and every original source/quality/resource rule.
Do not resume or install the failed third-update checkpoint.
