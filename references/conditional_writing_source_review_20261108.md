# Reliable rendering still requires learning what to write

Primary-source review, 8 November 2026. This review informs the unresolved
free-writing component. It adds no local model result, optimizer update,
checkpoint or replacement experiment. The registered contextual-learning
comparison remains unchanged.

## Eruku: optional style transcription is not optional answer content

[Eruku](https://arxiv.org/pdf/2510.23240), WACV 2026, predicts continuous VAE
writing latents with an autoregressive encoder-decoder. Its encoder receives
the text to generate, encoded with ByT5, plus style information. Synchronization
and explicit stopping symbols and text classifier-free guidance improve
content adherence. Dropping the **style image's transcription** leaves the
**generation text** available. The paper trains on synthetic English writing
and evaluates styled-text datasets; those results do not establish Chinese
question answering.

The [author project page](https://aimagelab.github.io/Eruku/) exposes the same
generation-text/style distinction. A demonstration of styled rendering must
not be described as independently answering an image question. Its treatment
of stopping and long outputs is relevant to a writing component; its supplied
answer content is outside this project's autonomous deployment boundary.

**Inference for this project:** using a teacher-supplied answer during training
is allowed. At deployment, an image-conditioned actor must itself infer the
content supplied to its writer. Replacing that inference with the reference
answer, a text-generating service, or a transcription of the required answer
would change the task. Removing an encoder's supplied-content input does not
establish that the remaining network has learned to supply it. Transfer would
need a new, independently assessed intervention, not merely a downloaded
rendering checkpoint. No Eruku experiment or download has been performed here.

## V2Flow: vocabulary compatibility is an explicit architectural choice

[V2Flow, sections 2.2 and 2.4](https://arxiv.org/html/2503.07493v1), maps visual
features to soft distributions over a fixed pretrained language vocabulary,
then quantizes them. Its image-generation language model adds visual codebook
entries and predicts their indices autoregressively; a flow decoder renders
the resulting sequence. Its reported reconstruction and text-conditioned
image-generation experiments are not image-question/image-language-answer
tests. The paper reports LLaMA2-7B training on approximately six million
image/text pairs for ten epochs, with global batch 1,024. Those are author
settings, not a measured local single-4090 budget.

**Inference for this project:** sharing a pretrained embedding geometry may be
a useful initialization hypothesis. This particular implementation nevertheless
uses vocabulary and categorical prediction machinery. Its image reconstruction
scores cannot establish an accurate answer distribution in a vocabulary-free
actor. The present codec already represents supplied answers accurately while
conditional prediction fails; another codec's fidelity alone does not answer
that failure. No local V2Flow execution or weight availability is claimed.

## D-AR: an image-generation bridge does not establish language learning

The [official D-AR repository](https://github.com/showlab/D-AR) describes
sequential visual token prediction with a Llama backbone and a diffusion
tokenizer. Its listed released generator results concern ImageNet 256-square
images; the page describes text-to-image work as ongoing. These are source
reports, not project measurements or evidence of reading and writing natural
language. No package or model weights were downloaded in this review.

## Decision for the active work

These sources do not supply a verified replacement for the failed conditional
writer. They distinguish three experiments that must not be conflated:

1. Render a supplied answer faithfully.
2. Predict an appropriate answer from the prompt.
3. Do both autonomously through the intended image interface on unseen cases.

The completed [readout comparison](decoder_metric_v82_result.md) already shows
that supplied-code fidelity does not qualify conditional writing, and the
[source-pixel component](contextual_pixel_spans_result.md) shows that new-string
copying does not qualify held-out answer selection. The running matched
comparison tests a reader intervention with the original-pixel motor fixed.
Its final result must determine whether that intervention improves the reader;
it cannot qualify absent-answer synthesis.

The immediate action remains to complete and inspect that registered result.
A subsequent writer treatment must state which failed conditional path it
changes, preserve the relevant earlier controls, and generate unmodified answers
whose wording is unavailable as an input crop. Selecting another model from
rendering examples, FID or reconstruction loss alone is not a justified remedy.
No new independent-ILM, efficiency, historical-script or novelty claim follows
from this source review.
