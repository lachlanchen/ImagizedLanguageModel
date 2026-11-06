# Decoder geometry as a possible training preconditioner

Research note, 7 November 2026. This candidate is motivated
by the [retained pixel-margin measurements](readout_pixel_margin_v81_result.md).
Its [decoder-only implementation and derivative checks](decoder_pullback_v82_preflight_result.md)
are now complete; autonomous learning under this objective remains untested.
The prior readout experiment and its quality criteria remain unchanged.

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
The measured decoder-only costs do not establish complete-actor efficiency.

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

Another recent preprint, Ahamed et al.'s
[Preconditioned Flow Matching, version 3](https://arxiv.org/html/2603.02337v3),
derives conditioning-dependent regression convergence in Gaussian and
Gaussian-mixture settings and tests reversible coordinate preconditioning.
That changes the intermediate distributions. The proposed decoder-sensitive
error instead reweights residual directions; it has no demonstrated
isotropization or convergence-rate guarantee.

Jiang et al.'s [TextLDM](https://arxiv.org/html/2605.07748v1) reports that strong
reconstruction alone does not yield effective diffusion representations, and
uses alignment to a frozen language model. Its VAE consumes discrete text
tokens and is not an image-only ILM implementation. Its result motivates
keeping semantic organization distinct from pixel reconstruction, rather than
assuming a more accurate decoder supplies linguistic structure.

The routes address different problems: physical sensitivity of the output
coordinates and semantic organization of the representation. Neither supplies
missing linguistic evidence or establishes generalization. The immediate
readout repair [fails](adaptive_readout_v81_result.md). The separately specified
physical-metric comparison tests an optimization hypothesis on the same
cohort; it makes no successful language claim before autonomous verification.
