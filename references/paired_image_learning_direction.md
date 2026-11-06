# Next learning test: adapt a visual generator with paired writing images

The [completed zero-shot baseline](image_edit_baseline_result.md) produces
copied instructions and corrupted pages, not the requested answers. It
performs no optimizer updates. Its failure does not establish whether the
visual generator can learn an instruction-to-answer mapping from paired
images. The next experiment must test that learning directly.

Use the undistilled [FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
source at revision `a3b4f4849157f664bdbc776fd7453c2783562f4d`. The inspected
transformer payload is 7,751,109,744 bytes, SHA256
`e109674697ffa1a3983126e32512f5428a9442bd8df59f9c95566ee90a473bb6`.
The base release's VAE configuration and weight bytes match the already
verified shared codec, so reuse those files rather than duplicate them.
Asset acquisition is preparation, not training or an improvement in answers.

The [author-published LoRA guide](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora)
uses the base checkpoint for learning and the four-step distilled model for
fast inference. This motivates the source choice; its style/editing examples
do not establish language learning or this project's training memory budget.
The first neural work must measure real forward/backward memory and gradients
on the actual paired-image path before a sustained optimizer run.

For input page $X$ and answer page $Y$, use the source image encoding,
patchification, batch normalization, packing and spatial coordinates. Writing
their encoded features as $z_X,z_Y$, the native noise-time convention is

\[
 z_\sigma=(1-\sigma)z_Y+\sigma\epsilon,\qquad
 \mathcal L=\mathbb E\|v_\theta(z_\sigma,\sigma,z_X,c)
                         -(\epsilon-z_Y)\|^2.
\]

Inference follows the source scheduler from noise toward decreasing sigma
and decodes the sampled image. The answer page is absent from inference.
The generic condition $c$ stays identical for every training and evaluation
example; no varying caption or answer embedding may bypass the visual input.
This is conventional conditional flow matching, with no new convergence,
sample-efficiency or language-understanding theorem claimed.

Train the real image-conditioning/generation path, initially through limited
adapters while retaining the pretrained visual prior. Use natural Chinese
question/answer pairs and fresh literal-copy strings as writing supervision.
Keep article-disjoint language evaluation and new copy strings separate from
optimization. Specify the corpus, splits, budget and acceptance criteria
before optimizer execution; a hardware preflight is not a language result.
The usable training interface should need source assets, data and device
configuration, without requiring a chain of obsolete experimental checkpoints.

Local training must be followed by autonomous complete answer images,
new-string copying and held-out question answering, including two questions
about the same passage and question-erasure controls. Calibrate any offline
OCR and inspect the raw outputs. Neither small-set memorization, copying a
prompt, a loss decrease nor a supplied-answer rendering can qualify the full
ILM objective. Historical forms, broader language, robustness and quality-matched
efficiency remain outstanding. The original general ILM goal stays active.
