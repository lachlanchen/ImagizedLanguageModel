# Separating visual readout from the language clock

The measured first-image failure and continuation failure are different
problems. A model can retain an answer's first-character signal but fail to
write it; it can also write the first form correctly and then advance its
context in the wrong way. A reliable image-language model must solve both.

This note describes the current construction and its limits. It does not
claim a new physical law or establish a successful language-learning method.

## What a visual representation can change

For a finite string-valued random variable T and an injective deterministic
renderer R, every possible string has exactly one corresponding image in the
rendered support. Consequently

\[
H(R(T))=-\sum_t p(t)\log p(t)=H(T).
\]

This elementary identity concerns discrete entropy, not differential entropy
of arbitrary continuous images. Rasterization can change locality, parameter
sharing, robustness and computational cost; it does not erase the information
that must be learned. If low resolution merges distinct strings, the rendering
instead loses information. This does not rule out a more efficient visual
architecture, but such efficiency must be measured at comparable language
quality, including rendering/encoding and pretraining costs.

Unobserved glyph meanings are also not uniquely determined by their shapes.
Relabelling those meanings leaves an unlabelled collection of glyph images
unchanged. Corpus use, translations, aligned explanations or other grounding
must supply the missing relationships. Visual similarities can provide useful
statistical clues without guaranteeing that an unseen historical script is
automatically readable. This is a motivation for learning from actual books
and aligned historical forms, not for a fixed runtime character inventory.

## A source-unit decision is not a glyph distribution

Let the offline source model predict native units v with probabilities
p(v | h), and let psi(v) be the first complete glyph of a unit. The induced
first-glyph distribution is

\[
p(g\mid h)=\sum_{v:\psi(v)=g}p(v\mid h).
\]

In general, taking the first glyph of the most likely source unit differs
from taking the most likely first glyph:

\[
\psi(\arg\max_v p(v\mid h))\ne\arg\max_g p(g\mid h).
\]

For a mathematical example, give the three units 语言, 语法 and 文字
probabilities .30, .29 and .41. The most likely unit starts with 文, while
the combined probability of the first glyph 语 is .59. This illustrates an
aggregation distinction; it is not a model experiment or an accuracy result.

The retrospective audit uses the first expression as an offline diagnostic.
It shows a readout gap, but it does not measure the true glyph distribution,
complete-answer competence or a deployed lexical policy. Likewise, forcing a
synthetic vector to have a particular winning source row does not establish
that the vector resembles an actual image-conditioned language state.
The [failed broad transfer](glyph_first_function_transfer_result.md) tests one
such synthetic recipe. Its failure is preserved rather than interpreted as
proof that all possible visual readouts are impossible.

## Writing and advancing context are separate operations

The deployed model has a visual reader V, contextual body F, continuous writer
G, image decoder D, image-feedback encoder E and learned stopping head S:

\[
h_0=F(V(P)),\qquad
\hat X_t=\operatorname{threshold}(D(G(h_t))),\qquad
h_{t+1}=F(E(\hat X_t);\text{past}_t).
\]

The actual emitted pixels, including errors, enter E. A reference glyph,
character ID, font lookup or teacher embedding cannot replace that feedback
while claiming the same autonomous execution.

Suppose a native source step would emit the two-character unit 语言. Emitting
only 语 and advancing once does not reproduce that native step: the intended
remainder 言 has not been emitted. A character-time model needs learned
dynamics that preserve the relevant state at the finer writing clock. It
cannot obtain that equivalence merely by deleting the source output matrix.

The source-only feedback controls expose this distinction on reused images:
native-unit feedback yields 12/16 literal matches, while exact-character and
image-encoded character feedback each yield 3/16. These routes change both
representation and advancement, so the difference does not isolate a single
cause. A separate fixed-line-framing attempt also fails. More decisively,
first-image damage is already present *before any generated feedback*; clock
drift therefore cannot explain every failure.

## Pixel fitting does not guarantee a valid next form

The current image loss balances foreground and background pixel BCE. Under
an unrestricted marginal predictor its optimum describes pixel marginals.
Thresholding marginals need not choose a supported complete image. For an
elementary counterexample, images (1,0) and (0,1) each have probability .5;
thresholding both marginals at .5 emits (1,1), which has probability zero.

The actual neural decoder shares weights and a common latent across pixels;
this example does not prove that it will produce that failure. Nor is marginal
loss the sole established problem: earlier normalized joint-pixel and
autoregressive output laws also fail the language tests. The useful separation
is between physical writing fidelity, correct conditional selection and stable
continuation. Each requires its own observed-image assessment.

## The present and subsequent learning tests

The [natural first-image experiment](natural_first_image_protocol.md) trains
the writer on actual complete-image prefills, with paragraph-level calibration
holdouts. Its cached final states are valid for fitting G because the reader
and contextual adapters remain frozen. It cannot by itself establish grammar,
meaningful complete answers, reliable stopping or arbitrary-script support.

If that physical test succeeds, a subsequent continuation intervention must
learn from actual image histories and evaluate complete unrepaired answers.
Changing contextual parameters invalidates a cache of their final states.
The existing exact frozen-prefix construction instead retains

\[
H_{32}=F_{0:31}(V(P),E(X_{<t})),\qquad
h_t=F_{32:35}^{\Delta}(H_{32},\Pi),
\]

with every prefix position and rotary coordinate Pi. Gradients through the
trainable tail then remain exact with respect to that frozen prefix. Using
generated histories still requires honest sampling/alignment accounting; it
does not automatically provide an unbiased policy-gradient estimator or
guarantee language improvement. The completed adaptation gives only partial
answer transfer, and the stronger first-image fit still needs verification.

Learning on model-visited states has an established precedent in
[DAgger](https://proceedings.mlr.press/v15/ross11a.html). Its guarantees require
the stated imitation-learning and no-regret assumptions. They do not transfer
automatically to an image writer: a reference next glyph after an erroneous
image prefix must be a valid supervisory action, and the losses and trajectory
sampling must match the analysis. Replaying damaged prefixes alone is not a
proof of a DAgger guarantee for this model.

The full objective remains an independently usable image-input/image-output
language model. A first-glyph gate, a small answer pilot or the inherited
7.522-billion-parameter body's ability is insufficient to declare it achieved.
