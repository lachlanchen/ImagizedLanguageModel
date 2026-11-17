# Candidate: preserve the writer and strengthen its conditional computation

The [paired image consequence](reader_writing_consequence_result_20261118.md)
shows that reducing reader coordinate error from .2695 to .2281 does not improve
the frozen actor's writing. Earlier full-context adaptation, predictive KL,
joint physical-image likelihood, mixtures, question assignment and native
source-state controls also fail their respective language or physical tests.
Do not repeat those recipes or infer an independent-pixel limitation for the
current conditional flow. This candidate changes one part of that flow: where
its language condition enters the computation.

The current compact writer adds a projected contextual state and time once,
before three ordinary transformer layers. Those layers receive no other
conditional modulation. This is a source-code observation, not proof that
they ignore the prompt. The negative image consequence motivates testing this
remaining interface rather than assuming that another coordinate reduction
will repair its output.

Retain the original reader, context, codec, count and length/stop predictors.
Compare a common explicit execution of the original writer with a copy that
adds scale and shift after each existing transformer layer:

\[
 c=C(h)+T(t,t^2),\quad x_0=P(z)+p+c,\qquad
 u_\ell=B_\ell(x_\ell),\qquad
 x_{\ell+1}=u_\ell\odot(1+a_\ell(\operatorname{LN}(c)))
                 +b_\ell(\operatorname{LN}(c)).
\]

Initialize every new affine map to zero. In a common execution path this
preserves the old velocity function at initialization, while allowing its
conditioning to change with fitting. Keep the final normalization and output
projection. The existing full-width velocity path remains; this is neither a
width intervention nor a new image output law. No new character inventory,
lexical head, lookup, reward judge or source embedding enters generation.

[Peebles and Xie](https://arxiv.org/abs/2212.09748) establish adaptive
conditioning as a useful design choice for diffusion transformers in image
generation. Their [official code](https://github.com/facebookresearch/DiT/blob/main/models.py)
uses adaptive normalization and residual gates. The proposed identity extension
of an already fitted writer is different from initializing a fresh DiT block;
their results do not qualify this ILM. Its benefit is a hypothesis. No theorem
guarantees useful language or one-GPU frontier-model parity.

First implement and verify function preservation, conditioning gradients,
valid-position masking, finite fitting and exact reset. Then separately bind
a real training-only comparison on the same three complete records, using
the original final actor and matched sampling/training noise. Freeze reader,
context, codec and stopping. Update both writers under the same flow objective
and budget; preserve every actual generated tile. These support records do
not qualify generalization. Do not substitute a successful subset, a longer
fit or a changed sampling schedule after observing the result.

Require an actual image improvement over both initialization and unchanged
conditioning, with no additional whitespace errors, before considering a
larger comparison. A falling flow loss is insufficient. A failed bounded
comparison stops this intervention; it does not establish an impossibility
theorem. A pass would require a new full-source registration and the unchanged
complete-language evaluation, including actual emitted feedback and learned
stopping. No full fit is admitted by this design note. All earlier checkpoints,
failed reader gates and autonomous scores remain unchanged.
