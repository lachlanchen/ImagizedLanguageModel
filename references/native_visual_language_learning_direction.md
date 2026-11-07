# A visual-access repair followed by native continuous writing

This route addresses a measured interface failure. It is not a working ILM,
a novel foundation architecture or a claim of parity with an 8B language model.
The [source observations](native_unified_source_result.md) retain both successes
and failures. The image path produces wrong copying answers, while the same
body recovers all eight copy contents from original text, with unwanted quotes
on four Chinese responses. Readable VAE reconstructions preserve the physical
marks. Language knowledge and access through this visual path are therefore
separate practical questions. The evidence does not select a unique cause.

## Preserve the interfaces that already have useful structure

The native source contains a continuous codec E,D, semantic image frontend R,
shared contextual body H, a language head U and an image velocity head G.
The complete source has 2,830,777,056 parameters plus a 126,892,531-parameter
codec. Its 1.5B label refers to the language body, not this complete footprint.
All these source components retain external pretraining.

Learn gated residuals A only on the clean source-image frontend. For image X,
the reading state is R_A(E(X)). The noisy answer-image branch retains its native
base frontend R_0. Disabling the gate restores the base computation. This
separates a repair of visual access from changing the image-generation prior.

The first optimization uses native language-head supervision:

\[
 L_{\mathrm{access}}=-\sum_j\log\operatorname{softmax}
 \{U H[C,R_A(E(X)),y_{<j}]\}_{y_j}.
\]

C is the same generic task control for every example. The changing question
enters through pixels; only preceding answer labels enter teacher forcing.
U and the language body are frozen. Unicode/token labels are explicit auxiliary
training supervision. They are not a claim of vocabulary-free training or
image-answer generation. Natural QA and new random copy strings are both used,
so accurate raster reading and prompt-dependent answer selection must be checked
separately from a loss decrease. Correct target history is absent during
autonomous readout evaluation.

The [specified visual-access experiment](native_visual_access_learning_protocol.md)
uses all 384 training examples and excludes 64 validation examples. Its real
two-update preflight trains 8,489,984 residual parameters across 158 native
linear layers at 6.899 GiB allocated peak. Gradients reach every up/down residual
on the second step, the zero-residual prefix matches, and the disabled prefix
is unchanged after the updates. These checks qualify execution only. The
completed [4,096-update experiment](native_visual_access_result.md) gives 27/32
exact stopped held-out copies and 0/32 correct natural answers. In paired
controls, 15/16 question pairs receive the same answer and removing the question
leaves 29/32 normal outputs unchanged. This recipe improves short copying but
fails page-level question selection. Visual access is not generally repaired;
the image-output stage is not admitted on these results. A subsequent learning
experiment must test better visual-language alignment and conditional selection
before treating this frontend as a useful language reader.

## The required image-output stage

After visual access is demonstrated, the changing question must continue to
enter through its raster. The output must come directly from the native image
velocity path. For target image Y and noise e, its native time convention is

\[
 z_t=(1-t)e+tE(Y),\qquad
 L_{\mathrm{write}}=\mathbb E\|G(H[C,R_A(E(X)),R_0(z_t)],t)
                                  -(E(Y)-e)\|^2.
\]

Use the native interleaved attention structure: source-image positions cannot
attend to future answer positions, and answer-image positions can attend to
the complete source and their own denoising frame. Different input/output
geometries require independently encoded visual regions and their correct
native spatial coordinates; reshaping an answer-aware crop is not a substitute.
The clean input adapters must not silently modify the noisy-image path. A
subsequent paired-image learning protocol must specify the actual implementation,
budget, numerical source equivalence and autonomous image assessments before
its optimizer runs. This stage is not implemented or validated by the reading
preflight.

Fixed structural controls can be compiled into numeric vectors offline. The
eventual image-only runtime must then remove its language lookup/output head
and generate pixels by integrating G and decoding D. It must not decode a
teacher's answer into text and hand that text to a font renderer. Typed prompts
may be rendered into input images at the application boundary; optional OCR is
an output view, not the generator. The continuous image representation permits
forms beyond an encoded character inventory, but historical-form capability
still requires actual data and tests.

If native flow supervision still loses exact writing content, a separately
calibrated training-only recognition constraint is a possible next hypothesis.
It is not an established repair, and independent raw-image assessment remains
necessary. The immediate work is the specified visual-access experiment, not
an accumulation of untested losses or architectures.

The end requirement remains an independent image-input/image-output model that
answers held-out questions correctly and writes legibly on one 4090-class GPU.
Auxiliary text readouts, copying, cached execution and successful memory checks
cannot complete that goal. Efficiency comparisons require matched useful task
quality and disclosure of complete parameters and inherited pretraining costs.
