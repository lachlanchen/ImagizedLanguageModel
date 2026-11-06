# Trainable visual writing over a frozen prompt memory

Research note, 6 November 2026. This proposal was examined while native Chinese
training was still running. It is a candidate for investigation after the
completed native evaluation and its prediction diagnosis. It introduces no new
model result, training run, checkpoint selection rule or language-success claim.

Writing can be treated as a conditional policy whose emissions are image
blocks. The policy must select an answer, render its marks and continue from
the pixels it actually emitted. A useful model must satisfy all three
requirements. Accurate reconstruction from supplied image codes establishes
the rendering component; it does not establish answer selection or reliable
continuation. The earlier [training-fit diagnosis](corpus_fit_v75_result.md)
already found failures with correct history and supplied answer length.

The current [native model](native_chinese_v80_protocol.md) retains full prompt
keys and values. Its writing predictor receives a contextual vector, however,
rather than attending to that memory as the noisy writing coordinates change.
The first block uses the original reader state directly. Later blocks obtain
a new state after the previously emitted image is encoded. The 36-layer
contextual body and its existing attention adapters remain frozen; gradients
through that body still train the image-input encoder. This is a narrower
learning interface than an adaptable continuous-output decoder.

There is a relevant precedent in robot learning. [π0](https://arxiv.org/html/2410.24164v1)
combines a pretrained vision-language backbone with a trainable continuous-action
expert. Noisy action queries attend to cached observation information during
flow integration. This establishes a mechanism for coupling semantic features
to continuous outputs in robotics. Its robot results do not establish language
generation from glyph images, and its tokenized language inputs would require
replacement to meet this project's runtime contract.

The distinction matters when borrowing the mechanism. In π0, the noisy action
suffix participates in attention at each integration step. Our current
predictor has the form Cθ(zₜ,t,hⱼ): its contextual condition is fixed during
that block's 32 denoising steps. A learned initial read query and adaptable
history transition could improve the condition, but would still leave this
fixed-condition denoiser in place. A small expert that queries the complete
source memory as a function of zₜ would be a separate architectural change.
Repeating the entire 8B contextual body at every denoising step would also
change the computational cost substantially.

The official [OpenPi implementation](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L228-L274)
separates prefix prefill from suffix sampling and reuses the prefix cache.
Its configuration exposes separate adaptation choices for the backbone and
expert. The checked [README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md)
estimates more than 22.5 GB for its LoRA fine-tuning configuration, while its
PyTorch support section excludes LoRA training. Those statements concern that
implementation; they provide neither a drop-in training dependency nor a
memory guarantee for ILM. The pinned source receipt is
[recorded separately](evidence/continuous_writing_expert_literature.json).

[VLAFlow](https://arxiv.org/html/2607.01586v1) compares continuous-action training
objectives under a shared architecture. Its robot experiments find that
language supervision and future-state alignment affect transfer, and that
stopping action-loss gradients into the VLM can reduce performance. This
motivates examining representation learning alongside the output loss. Our
current implementation already differentiates through the frozen contextual
body into the input encoder, so it is not the same stop-gradient intervention.
The paper supplies no guarantee for our glyph codec, data or training budget.

A relatively small adaptation candidate is already available in the local
code. [ImageAdapterLinear](../ilm/visual_lm/grounded_image_actor.py) allocates
rank-8 adapters with scale α/r, α=16, on q, v and o projections. The
[native actor](../ilm/visual_lm/native_image_actor.py) currently requires their
outputs to be zero and freezes them. Reusing these allocated adapters would
allow the visual transition to learn while retaining the pretrained base
weights. It would still be a complete model of roughly 7.6B parameters, with
a smaller locally optimized subset.

Let K₀(X) be the prompt cache produced by the unchanged base reader. Write a
projection's adaptable weights as

\[
 W_{\psi,\Delta}=W_\psi+(\alpha/r)UV.
\]

Introduce a learned continuous start query q. With Zᵢ=E(Yᵢ), the candidate
writing conditions would be

\[
 h_j^*=F_{\psi,\Delta}
 \bigl(K_0(X),q,A_\phi(Z_1),\ldots,A_\phi(Z_j)\bigr)_{\mathrm{last}},
 \qquad j\geq0.
\]

The first image would then follow an actual trainable read operation over the
full prompt memory. The existing clean predictor, image decoder and learned
stop could use hⱼ* without a language vocabulary projection. Every generated
block would still be thresholded to actual pixels and re-encoded before the
next decision. This describes a proposed computation, not observed capability.

Cache validity is a necessary condition. Applying updated adapters during
prompt prefill would change K(X), making the current multi-update replay pool
stale. The candidate must instead define two explicit modes: base-only prompt
reading and adapted continuation. Under that definition,

\[
 \frac{\partial K_0(X)}{\partial\Delta}=0.
\]

This identity permits reuse of the frozen prompt cache across adapter updates.
It is a statement about the chosen computation, not a learning or convergence
theorem. The base reader's prompt-memory digest must remain equal to the
existing reference. The new post-query state need not equal the old h₀.

Implementation would need to preserve the following contracts:

- Adapter mode during checkpoint recomputation must match the original
  suffix forward pass. Temporarily disabling adapters for prefill must not
  change backward recomputation or leave a disabled continuation behind.
- Checkpoints must include the initial query and every optimized adapter,
  in addition to image input, writing and stopping parameters. Exact resume,
  padded-prefix exclusion and causal future exclusion need actual-model checks.
- The additional query consumes a native position. Physical cache indices,
  original rotary offsets and generated-history positions must advance
  consistently in both full-history training and incremental inference.
- Runtime generation receives prompt pixels and emitted-image history.
  Reference codes and lengths remain restricted to training and assisted
  diagnosis. The original inference and physical-image requirements remain.

The adapter parameter count can be computed from the actual projection shapes:

\[
 P_\Delta=r\sum_{\ell,p\in\{q,v,o\}}
 \bigl(d_{\mathrm{in}}^{\ell,p}+d_{\mathrm{out}}^{\ell,p}\bigr).
\]

Optimizer state, parameter gradients and activation memory must be measured
in a real single-4090 preflight. Existing allocated adapters avoid a second
foundation-weight copy; that does not imply zero additional training memory.
The present noise-at-zero/data-at-one convention must also be retained
explicitly: the pinned OpenPi sampling code uses the opposite time direction.

The next intervention should depend on the completed
[native fixed-cohort diagnosis](native_corpus_fit_v80_protocol.md). Poor
supplied-code reconstruction would first implicate the codec. Poor prediction
with correct history would justify investigating conditional readout and
representation adaptation. Strong correct-history prediction with poor free
generation would instead make generated-history alignment a priority.
Successful training-set output with poor development output would leave a
generalization problem. These outcomes must not be collapsed into one cause.

If transition adaptation is tested, a matched frozen-transition continuation
should receive the same data, draws, objective and optimization budget. Adding
both an initial query and adaptation is a composite intervention unless their
effects are subsequently separated. The existing uniform time law and sampler
should remain fixed initially; the failed [time-coverage repair](writing_times_v76_result.md)
provides no basis for assuming that a different schedule solves the language
problem. Any new experiment requires its own prospective protocol, preserved
baseline, actual outputs and evaluation. The general ILM objective remains open.
