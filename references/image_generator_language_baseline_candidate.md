# Candidate baseline: language instructions through an image editor

Research assessment, 7 November 2026. This candidate has not generated an
answer or received local training. It does not replace the running V82
comparison or establish a correct learning direction.

The target behavior is a conditional image distribution: a page containing a
question enters, and a new page containing an answer leaves. A pretrained
image editor already supplies a joint image-conditioning path and a trained
image generator. Testing that route would separate reuse of an existing
generative visual prior from the current construction, whose continuous
writing interface is learned around a contextual language foundation.

The official [FLUX.2-klein-4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
describes image editing with a four-billion-parameter flow transformer and
reports an approximately 13 GB inference footprint. Its
[base checkpoint](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
is an available training source. An author-published
[paired-edit LoRA guide](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora)
reports a 24 GB consumer-GPU training route. These are upstream reports, not
local measurements or evidence of language capability. The card explicitly
warns that rendered text can be distorted and that the model does not provide
reliable factual information. Changing the adaptation data does not remove
those limitations without actual evidence.

The stock [pipeline configuration](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/raw/main/model_index.json)
includes a Qwen3 text encoder and tokenizer. Therefore its normal text-prompt
interface cannot itself serve as evidence of image-native language input.
The locally installed Diffusers 0.39.0 implementation of
`Flux2KleinPipeline.encode_prompt` accepts supplied `prompt_embeds` and calls
the text encoder only when they are absent. Its image branch separately
encodes reference-image pixels. This is source inspection, not a successful
encoder-free runtime test.

For a genuine image-input baseline, compute one fixed generic task condition
offline, freeze it, and use exactly the same numeric condition for every
question. All changing instructions, question content, passages and glyphs
must enter through the image. Exclude per-question captions, OCR transcripts,
answer-aware crops and per-question text embeddings from runtime. Do not
silently substitute a text encoder's answer for generated pixels. Verify a
fresh runtime without loaded tokenizer or text-encoder modules before
claiming that interface. Constant source-derived controls and all external
pretraining remain attributed dependencies, just as in the present actor.

If c is that fixed condition and E, D are the source image codec, paired
training would take an input page q and an answer page a and model

\[
 z_q=E(q),\quad z_a=E(a),\quad
 z_t=(1-t)\epsilon+t z_a,\qquad
 \mathcal L=\mathbb E\|v_\theta(z_t,t,z_q,c)-(z_a-\epsilon)\|^2.
\]

This is a conventional conditional flow objective, not a new learning theorem.
Here time runs from noise at zero to clean data at one; any implementation
must consistently convert the source scheduler's direction and time scale.
At deployment, a never enters the call: sample noise, condition on q and the
fixed c, integrate the declared source flow and decode the resulting answer
image. The generator's existing pretraining, rather than an untrained output
head, supplies its initial writing prior. Whether it can learn accurate
Chinese question answering through that image branch remains an experiment.

The first decision would be codec fidelity on the actual writing sizes.
Measure supplied-image encode/decode at fixed page geometries before training
language; a codec that destroys strokes cannot be credited as a successful
writer. The source [VAE configuration](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/raw/main/vae/config.json)
uses continuous image latents, but their existence does not guarantee
readable small print or faithful historical forms. Retain actual decoded
pixels and any failed cases; do not snap them to a font or character bank.

Subsequent evidence must separate new-string copying, seen-question fit and
held-out prompt-dependent answering. Different questions about the same
passage and question-blanking controls are necessary to distinguish an answer
from a copied or generic page. Hold out articles and content, document font
and layout changes, and retain every uncorrected result. These are requirements
for evaluating the candidate, not completed tests or permission to weaken the
current experiment's original gates. A copying success or attractive sample
alone cannot qualify language learning.

The shared LocalLLM installation currently documents Z-Image-Turbo as a
text-to-image service; it does not expose image-conditioned editing. Its API
cannot implement this experiment merely by changing a request. Its existing
Diffusers installation is reusable for source inspection, but no service,
environment or weight installation is changed here. Any later experiment must
first recheck available shared weights and resources, run sequentially after
the current GPU job, and report actual total parameters, codec cost, memory,
training time and language quality. Four billion generator parameters do not
include every pipeline component or the cost of external pretraining.
