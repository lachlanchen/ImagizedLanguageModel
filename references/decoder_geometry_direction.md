# Decoder geometry as a possible training preconditioner

Research note, 7 November 2026. This is an unimplemented candidate, motivated
by the [retained pixel-margin measurements](readout_pixel_margin_v81_result.md).
It changes neither the running readout experiment nor its quality criteria.

The fixed codec reconstructs supplied writing well, but coordinate error and
physical stroke error are not equivalent. The continued control has mean
first-block coordinate MSE 0.035 and 182 incorrect pixels; later correct-history
blocks have much larger errors. A useful objective should respect the decoder
without confusing a better physical fit with language understanding.

Let R map two continuous tile codes to the 2,048 decoder logits, including
the actual per-tile normalization. For noisy code x and a residual v, consider
the state-only metric

\[
 G(x,t)=\eta I+\alpha(t)J_R(x)^T W(x,t)J_R(x),\qquad\eta>0,\quad W\succeq0.
\]

Its quadratic error is computable as η||v||² plus
α||W^(1/2) J_R(x)v||² using a Jacobian-vector product; the full matrix need not
be stored. Added compute would involve the small fixed decoder, but actual
time, memory, derivative support and gradient stability must be measured.
No implementation or efficiency result is claimed here.

For clean target z, condition on x,t,h and minimize the expected quadratic
loss of c−z in this metric. If G is fixed given those observed variables and
independent of the predicted c and hidden target z, then

\[
 \nabla_c\,\mathbb E[(c-z)^T G(c-z)\mid x,t,h]
 =2G(c-\mathbb E[z\mid x,t,h]).
\]

Positive definiteness therefore preserves the conditional-mean optimum of
the original squared loss. This elementary conditional-risk argument concerns
the function-space optimum, not finite-model convergence or language ability.
Target-dependent class weights or evaluating the metric at the predicted
endpoint do not satisfy the stated assumption. A nonlinear pixel BCE term
also needs its own analysis; it cannot inherit this conclusion automatically.

Related work already treats geometry explicitly. Chen and Lipman's
[Flow Matching on General Geometries](https://arxiv.org/html/2302.03660v3)
constructs manifold flows and uses their metric. The candidate above keeps
the existing linear noisy-code paths and only reweights their regression;
it is not a reproduction of their geodesic-flow construction.

A recent alternative, Nützel et al.'s
[Semantic Latent Alignment for Flow Matching with Invertible Bi-Lipschitz Adapters](https://arxiv.org/html/2610.21332v1),
fits a constrained coordinate transform to an offline teacher while keeping
the codec fixed. Its evidence concerns medical images, not writing or language.
For ILM, teacher information, geometric distortion and numerical inverse
accuracy would each need testing. Exact coordinate invertibility alone does
not guarantee unchanged thresholded glyph pixels after approximate inversion.

The two routes address different problems: physical sensitivity of the output
coordinates and semantic organization of the representation. Neither supplies
missing linguistic evidence or establishes generalization. The immediate
decision remains the completed matched readout result; any subsequent loss or
coordinate change requires a new bounded experiment, not another uncontrolled
long training run.
