# An invariant noise subspace in the compact writer

The compact writer uses 768-dimensional codec vectors and a 384-channel
Transformer. Its final operation is a linear velocity projection. The retained
4,049-update checkpoint has an output matrix of shape 768 × 384 and measured
rank 384. This restricts the velocity in a way that training cannot remove.
The [matrix evidence](evidence/compact_visual_flow_rank_20261111.json) uses
actual learned weights and the frozen codec, with no neural generation.

Let the writer's velocity at a glyph position be

\[
 v_\theta(z,t,c)=W h_\theta(z,t,c)+b,
 \qquad W\in\mathbb{R}^{d\times m},\quad d=768,\ m=384.
\]

For an orthogonal projector \(P\) onto the complement of the columns of
\(W\), \(PW=0\), so \(Pv_\theta=Pb\). The eight-step sampler starts with
\(z_0\sim\mathcal N(0,I)\) and adds the velocity with increments of \(1/8\).
Its endpoint therefore satisfies

\[
 Pz_1=Pz_0+Pb.
\]

This holds for every hidden trajectory, including every conditioning context.
The exact-arithmetic flow cannot contract or condition the initial Gaussian
variation in these 384 directions. Floating-point rounding can perturb this
identity; the audit measures the linear model in float64, not a hardware-level
identity of the complete sampler.

The training target is \(y-\epsilon\), with \(\epsilon\sim\mathcal N(0,I)\)
independent of the clean latent \(y\). With a fixed output matrix and bias,

\[
 \begin{aligned}
 \mathbb E\|v_\theta-(y-\epsilon)\|^2/d
 &\geq \mathbb E\|P(y-\epsilon-b)\|^2/d\\
 &= \{\mathbb E\|P(y-b)\|^2+\operatorname{tr}P\}/d\\
 &\geq (d-\operatorname{rank}W)/d=0.5.
 \end{aligned}
\]

This is a lower bound on expected loss with fresh noise for the continuous
linear-output model. It is not a minimum for every finite training batch or
a measured language score. The bound applies at any sampled flow time.
Cross-position attention does not change the final per-position projection.

The float64 projection annihilates the learned matrix to maximum absolute
error 6.41 × 10⁻¹⁶. Two arbitrary eight-step hidden trajectories preserve the
endpoint identity to 3.78 × 10⁻¹⁵. The fixed decoder's first 3,072 × 768
projection transmits 49.01% of its full isotropic-noise energy from this
complement. That is a decoder preactivation measurement; the nonlinear
decoder's pixel sensitivity and answer readability still require evaluation.

The feedback encoder also ends in a 2,048 × 384 linear map. Its outputs occupy
an affine space of dimension at most 384. This audit does not measure the
approximation error that restriction imposes on its source targets. Treat
that separate question as unmeasured.

## Consequence for the experiment

Complete the registered five-checkpoint image evaluation with the unchanged
model. Preserve every raw output and compare the actual answer behavior.
Additional epochs alone cannot remove the writer's invariant component in
the continuous model. The next writer should provide a full-dimensional
velocity path. A direct bounded repair is to set its input and hidden width
at least to the 768-dimensional latent width and verify the resulting output
rank. Keep the encoder, codec, contextual foundation, corpus, optimizer and
evaluation fixed for the first comparison. Report the extra parameters,
compute and all initialization differences. A full-rank head still needs
actual training and language evaluation; it is not sufficient by itself.

Preserve the present implementation and checkpoint loader as the baseline.
The audited run and its queued evaluation sources remain unchanged. The
CPU matrix calculation took 0.276 seconds at 1.35 GiB peak process RSS and
initialized no CUDA context.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  python scripts/audit_compact_visual_flow_rank.py \
    --checkpoint artifacts/compact_visual_span_fit_v1/image_heads_step_0004049.pt \
    --output artifacts/compact_visual_flow_rank_reproduction
```
