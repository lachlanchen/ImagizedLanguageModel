# Writing occupies little area but already receives substantial error

One actual saved native denoising forward assigns **21.34% of squared error**
to the 0.293% of latent cells spatially aligned with target ink. Their mean
error is **92.31 times** the mean outside those cells. Small ink area alone
therefore does not establish that the loss neglects writing in this draw.
This finding supplies no answer-quality or causal explanation for the
separate ongoing training run.

The [numerical audit](evidence/native_flow_spatial_loss.json) reopens the real
prediction and target velocities from the completed one-update preflight.
It reproduces its unweighted native MSE, 0.0157170743, at sigma 0.6328125.
The native target contains 946 black pixels in the full 1024-square page.
No model, new encoder, sampler or optimizer is run by this audit.

| Spatial region in the saved latent grid | Grid area | Squared error | Squared prediction gradient |
| --- | ---: | ---: | ---: |
| Any target ink within each corresponding 8×8 image block | 0.293% | 21.337% | 21.339% |
| That region expanded by one latent cell | 0.513% | 25.302% | 25.302% |
| That region expanded by three latent cells | 1.099% | 28.764% | 28.766% |
| Fixed 512×256 answer region | 12.500% | 41.774% | 41.774% |

For the squared velocity error, with D scalar coordinates,

\[
L=\frac1D\sum_i(p_i-u_i)^2,\qquad
\frac{\partial L}{\partial p_i}=\frac2D(p_i-u_i).
\]

In exact arithmetic, a region's share of squared prediction-gradient norm
equals its share of squared error. Actual CPU autograd through the native
loss also measures the BF16 gradient rounding shown in the table. These
gradients are with respect to the saved prediction tensor. The network's
Jacobian mixes spatial positions, so the audit does not determine how much
each region contributes to an adapter update.

The spatial masks are descriptive alignments. The codec's receptive field
can spread writing information beyond an ink-aligned latent cell. The
measurement covers one fixed pre-adaptation example and one noise level;
it does not characterize the complete training distribution. In particular,
it does not demonstrate that foreground weighting would help or harm language
learning. The current registered objective and schedule remain unchanged.

Physical error allocation, image reconstruction and question-dependent answer
selection require distinct evidence. The complete prospective answer
assessment remains the test of the ongoing model's language capability.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_native_flow_spatial_loss.py \
  --output artifacts/native_flow_spatial_loss_v1
```

Use the existing model Python and a fresh output directory. All source tensor,
target-image, execution-report and implementation hashes are in the record.
