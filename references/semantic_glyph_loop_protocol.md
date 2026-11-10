# Continuous semantic glyph loop: staged repair protocol

The [matched reader baseline](pixel_language_source_reader_result.md) retains
useful visual question answering, while the learned pixel writer fails on
unseen answers. The repair preserves the reader's original assistant state
and gives each output event one complete 32×32 glyph. It separates learning
glyph formation from learning which glyph follows a contextual state.

Let E map actual glyph pixels to a continuous semantic vector, D map that
vector to pixels, T be the inherited visual contextual body, and G its new
continuous next-glyph readout. The runtime is

\[
h_0=T(I_{\mathrm{prompt}}),\quad
z_t=G(h_t),\quad I_t=\mathbf{1}[\sigma(D(z_t))\geq 1/2],\quad
h_{t+1}=T(E(I_t);\mathrm{KV}_{\leq t}).
\]

A separate learned stopping probability chooses termination without a
reference length. G and E operate in normalized semantic coordinates; a
global per-dimension affine transform restores the contextual input scale.
There is no learned initial writing query. Actual emitted pixels determine
feedback. The runtime contains no source vocabulary matrix, tokenizer,
character IDs, stored glyph images, font renderer, or external teacher.

This stage uses external pretrained language representations; it is not a
foundation trained from scratch, a proof of human reading mechanisms, or a
matched-quality efficiency advantage over a text model.

## Stage A: image/semantic interface

Select up to 4,096 characters from ASCII and descending character frequencies
in the training split of the prepared CMRC corpus. Development questions,
answers and helper hidden states are forbidden for training. The initial
anchor requires one exactly round-tripping source input unit and verified
font coverage. Record exclusions; do not silently drop later evaluation
cases containing unsupported characters. Use the locally available Noto Sans
CJK SC collection face, 26-pixel size on a 32×32 binary canvas with a common
baseline. This experiment tests one canonical font, not historical writing.

Offline preparation obtains source input embeddings and rendered images.
Identical bitmaps share their mean training anchor, avoiding contradictory
encoder targets. IDs and bitmaps stay in the training dataset, never in the
runtime checkpoint. Global affine statistics contain no character-indexed
table. A CNN encoder and convolutional decoder learn semantic MSE/cosine
alignment, direct reconstruction, and reconstruction through the encoder.
The decoder sees small semantic noise; the encoder sees one-pixel shifts
with white padding. Foreground/background-balanced BCE is a reconstruction
objective, not a normalized full language joint likelihood.

Run 6,000 batch-128 AdamW updates, learning rate 3e-4 with cosine decay to
one tenth, seed 202611101. Encoder/decoder parameters are FP32 with BF16
convolution/linear autocast. G and stopping parameters remain frozen and
untrained. Assess all prepared glyphs at zero and each 1,000 updates, retain
raw metrics and the complete latest checkpoint. Report per-glyph foreground
F1, exactness and semantic alignment. Seen-identity reconstruction or shifted
glyph recovery is not unseen-character generalization or language learning.
The runtime explicitly refuses to generate language from this primitive
checkpoint.

This separate small training stage uses the project's exclusive GPU0 lock,
requires 20 GiB GPU free and 64 GiB host RAM at entry, and caps allocation at
5 GiB and elapsed time at 1,800 seconds. Monitor at least every twenty updates
and every assessment batch: RAM below 32 GiB aborts; memory-pressure averages
above 0.5% some or 0.1% full pause further computation, with 300 seconds total
waiting allowed. Inspect project-owned processes before entry, including
when global swap exceeds 75%; do not stop another project's services. Global
swap occupancy is recorded, not altered. These are new bounded-stage rules;
the completed diversity training sources and admission rules stay unchanged.
Preserve failed attempts and their actual optimizer counts.

## Stage B: language continuation, still required

Only after the primitive works, freeze E and D. Obtain contextual states with
gold previous glyph **images** encoded by E, using training-only question
images and the native assistant state. For each state, train G toward the
next glyph's continuous anchor and rendered image, or the terminal stopping
event. Never feed the current target image before predicting it. A cached
frozen state is acceptable only when its inputs, causal position, bridge
checkpoint and source bindings are authenticated. No cached development
states or answers may enter training. Broader training text may provide
additional continuation examples; coverage must be reported honestly.

Evaluation must run autonomous actual-image feedback with learned stopping,
retain complete raw images, and include all fixed Chinese development cases
and erased-question controls. State the canonical font/anchor limits and
unsupported output forms. Inspect output images before source comparison,
report errors and caps, and distinguish literal pixel equality from semantic
correctness. Stage A alone cannot support a working-ILM claim. Stage B is not
yet implemented or validated in this protocol revision.

Pixel-space language generation has prior work, including
[PIXAR](https://aclanthology.org/2024.findings-acl.874/),
[PIXEL](https://arxiv.org/abs/2207.06991), and
[MIXAR](https://arxiv.org/abs/2604.11575). This repair is motivated by the
local matched failure analysis, not a claim to have invented pixel language
modeling or a new physical law.
