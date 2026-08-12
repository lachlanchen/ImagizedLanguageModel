# Visual Field Machine: A Non-Token Architecture for Visible Language

## Why the obvious design failed

The first implementation compressed a complete 384 by 384 answer page and
learned a conditional rectified flow from noise. Its losses improved, but the
generated page was unreadable. This is not merely a small-model problem. The
objective gives most pixels to paper, margins, and layout; exact strokes occupy
a small minority. It also asks one network to discover semantics, remember
facts, typeset prose, and reproduce historical evidence through the same noisy
channel.

The new design rejects three assumptions:

1. A language image should not be generated as an undifferentiated photograph.
2. Knowledge does not all need to be compressed into model weights.
3. An attested historical mark should not be synthesized when its source image
   can be copied exactly.

## The machine

The **Visual Field Machine (VFM)** has four interacting systems.

### 1. Retina: sparse seeing rather than square resizing

For input canvas \(X\), let \(Y\) be luminance. An ink-energy field is

\[
S(X)=\operatorname{pool}\left(\alpha(1-Y)+\beta\lVert\nabla Y\rVert_1\right).
\]

Non-maximum suppression chooses \(K\) locations \(c_i\). The model receives a
small peripheral image \(P(X)\) and full-detail crops
\(F_i=\operatorname{gridSample}(X,c_i)\). Thus thin strokes survive without
making the whole page expensive.

The continuous retinal state is updated by repeated content-directed reads:

\[
a_i^t=\operatorname{softmax}_i\left(
q(h_t)^\top k(f_i)/\sqrt d + \lambda\log(S_i+\epsilon)-\mu u_i^t
\right),
\]

\[
r_t=\sum_i a_i^t v(f_i),\qquad
h_{t+1}=\operatorname{GRU}(r_t,h_t),\qquad
u_i^{t+1}=u_i^t+a_i^t.
\]

The coverage term \(u\) produces inhibition of return: successive reads can
attend to different visible regions. All inputs to these equations are image
features and coordinates.

### 2. Visual episodes: knowledge that can be seen and amended

An episode is

\[
M_j=(k_j, V_j, p_j),
\]

where \(k_j\) is a continuous embedding of a prompt image, \(V_j\) is an answer
image or set of answer regions, and \(p_j\) is provenance. Retrieval uses visual
similarity \(j^*=\arg\max_j e(X)^\top k_j\). A new book page or correction can
be inserted by encoding its prompt image once; the neural model need not be
retrained.

This first proof intentionally behaves as an episodic visual reader. It is not
misreported as open-ended reasoning. It tests the necessary substrate: whether
an independently trained image encoder can address correct answer images under
new fonts, degradation, layouts, scripts, and prompt renderings.

### 3. Evidence gate: do not invent the archive

Every output region has one of three origins:

- **copy**: exact pixels from an attested source asset with provenance;
- **retrieve**: an answer or line image from visual episodic memory;
- **write**: newly rendered connective language from the neural ink path.

Historical glyph panels presented as evidence must use `copy`. A generated
historical-looking form may be shown only as a reconstruction and must never
inherit an attested label.

### 4. Motor canvas: deposit ink, not photographic noise

The generative stage operates on occupied line and glyph regions. A recurrent
canvas state predicts a continuous region proposal, mode gate, and line-field
latent. A high-fidelity decoder emits an ink alpha field plus colour; alpha
compositing places it on the page. Empty paper is a deterministic canvas and
does not consume generative capacity.

Long answers are recurrent visual continuation, not a fixed number of text
tokens. The state advances until an image-derived stop/occupancy head closes the
page or opens a continuation canvas.

## Learning without linguistic IDs

At the offline boundary, licensed text records may be rasterized. The student
then receives only images. For two visual renderings \(X_i^a,X_i^b\) of one
prompt and its answer image \(A_i\), the first-stage loss is

\[
\mathcal L =
\mathcal L_{\mathrm{NCE}}(e_q(X^a),e_q(X^b))+
\gamma\mathcal L_{\mathrm{NCE}}(e_q(X^a),e_a(A))+
\eta\mathcal L_{\mathrm{var}}.
\]

The first term learns typography/damage invariance. The second aligns questions
and their visible answers. The variance term prevents representational
collapse. No model call accepts strings, token IDs, code points, OCR output, or
class labels.

## Claims this design permits

- Exact rare-form output can be more reliable than generic image generation.
- Visual knowledge can be inserted and corrected without gradient updates.
- Computation can scale with occupied ink rather than page area.
- A small model can prove robust visual addressing and evidence-preserving
  answers on one RTX 4090.

## Claims it does not yet permit

- Episodic retrieval alone is not general language reasoning.
- Rasterized Alpaca does not make a small visual encoder equivalent to Qwen 8B.
- A synthetic ancient-looking glyph is not historical evidence.
- Efficiency and bounded parity require measurements, not intuition.

The path to an independent language model is therefore incremental but not
conventional: prove visual addressing, then learn continuous visual
composition, then recurrent visual reasoning, while preserving the same strict
image-only boundary throughout.
