# A visual copy distribution for grounded language

Research direction, 2026-11-05; not yet implemented or qualified.

V55/V56 separate two failures: a visual density can generate clear glyphs,
while a correct-looking continuation can still be irrelevant or repetitive.
The next task should require reading a natural question and identifying its
answer in a visible paragraph. It should preserve the free-writing density
rather than redefine the complete ILM as an extractive system.

## Proper image probability with a copy branch

Let H contain observed image cells X_i, reader state h and contextual source
features k_i. A learned attention distribution a_i(H) addresses positions in
the current visual input. A learned gate lambda(H) mixes copied observations
with the existing free image density:

    p(Y | H) = lambda(H) sum_i a_i(H) 1[Y = X_i]
               + (1-lambda(H)) p_density(Y | H).

This is normalized over binary rasters. Repeated identical source images
contribute summed copy mass, not competing symbolic labels. Training uses
the exact image probability with that marginalization. Source equality is
computed from pixels. No Unicode ID, transcript or persistent character bank
is an input or target to the student. Position attention addresses the actual
observed image, not a fixed language vocabulary.

The copy operation preserves an unfamiliar or unencoded form exactly when
the model selects it. The free branch still supports writing absent from
the prompt. Sampling chooses a branch, then an observed image position or the
learned pixel law. A decoding method must score its own emitted images under
the mixture, including summed mass when an image occurs more than once.

This applies an established pointer-generator idea to the native image
interface. [CopyNet](https://arxiv.org/abs/1603.06393) integrates copy and
generation in sequence learning; [See et al.](https://arxiv.org/abs/1704.04368)
combine source pointers with a generator and discuss factual copying and
repetition. Those are prior work. This note makes no priority claim for a copy
mechanism or for the broader pixel-language paradigm.

## What must actually be learned

The target image is not supplied to the model. A visible paragraph contains
many candidate forms; the question determines which answer sequence to
produce. The new CMRC subset supplies natural questions, unseen paragraphs
and 393 same-paragraph counterfactual pairs. Merely preserving a supplied
answer or picking a plausible glyph cannot pass that test.

Train an image-only question/passage/answer path and compare it with the same
reader/density without the source-copy mixture under matched data and budget.
Retain the natural-stream likelihood task so that extractive training cannot
silently eliminate the free-language path. Any prompt-length change, stopping
convention, loss, routing and optimizer choice must be fixed before training.
No coverage penalty, special answer-location rule or decoding repair should
be introduced without its own evidence.

The decisive outputs are raw answer images on unseen paragraphs, exact-image
and character-order measures, correct changes under paired questions, and
behavior with irrelevant or removed source context. All reference labels and
any optional OCR remain outside deployed inference. Font transfer and native
unencoded-form copying need separate tests before a historical-script claim.

The supplied-image V54 motor experiment revealed the intended raster directly.
The proposed source attention would instead learn the choice among all images
in the user's input. That distinction is essential. A correct visual copy
branch is a useful component; only measured question-dependent answers would
establish the next language capability. The full independent ILM goal remains
broader than paragraph extraction.
