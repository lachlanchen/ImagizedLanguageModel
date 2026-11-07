# Candidate audit: retain native reading and continuous image generation

Reviewed 7 November 2026. This original source-level candidate audit is followed
by the [completed native-source experiment](native_unified_source_result.md)
and a [visual-access learning protocol](native_visual_access_learning_protocol.md).
Neither establishes a working ILM or a proven repair. The complete
[paired-copying observation](paired_padding_result.md) fails reliable writing.
The existing [native contextual writer](decoder_metric_v82_result.md) also fails
even with correct answer history. Those failures do not identify one universal
cause or show that image-based language modelling is impossible.

## The interface problem

The paired-copying system preserves a trained image generator, but every sample
receives the same generic language condition. It cannot obtain the changing
question's meaning through that constant. All changing content must be learned
from the image branch; 512 updates on 32 copies do not constitute language
pretraining or QA training. The other system preserves a pretrained contextual
body but adds a new continuous writing interface. Its original language-output
competence is not automatically transferred to that interface.

A possible next initialization retains a backbone already trained with both
visual understanding and continuous image generation. This avoids assuming
that either newly attached interface inherits the missing capability. It does
not automatically solve text spelling, question understanding or image-answer
instruction following. No single-4090 training advantage follows from its name
or its language-body parameter count.

## Primary sources inspected

- [JanusFlow official implementation](https://github.com/deepseek-ai/Janus/blob/main/janus/janusflow/models/modeling_vlm.py)
  constructs a visual understanding encoder/aligner, a continuous visual
  generation encoder/decoder and a shared Llama body. The understanding path
  inserts projected image features into the language model's input embeddings.
  The [official examples](https://github.com/deepseek-ai/Janus#janusflow)
  separately demonstrate image-to-text understanding and text-to-image flow
  generation. They do not establish image-only questions answered in written
  answer images. A measured complete deployment count must include the visual
  branches and VAE rather than relying on the advertised model name.
- [Show-o2 official documentation](https://github.com/showlab/Show-o/tree/main/show-o2)
  specifies unified understanding and generation using autoregressive and
  flow-matching heads in a 3D causal VAE space. Its documented demonstrations
  include reading image text and generating images, with a separate HQ release
  described as improving text rendering. The [HQ configuration](https://huggingface.co/showlab/show-o2-1.5B-HQ/blob/main/config.json)
  identifies Qwen2.5-1.5B-Instruct, SigLIP, 16-channel image latents and ten
  diffusion layers. These component facts do not establish Chinese writing
  reliability or the complete parameter/memory footprint on this workstation.

These are existing unified architectures, not a new invention by this project.
The current source review is insufficient to select a successful checkpoint.
Model/weight licences, pinned revisions, native sampling, full parameter count
and actual image resolution remain acquisition prerequisites.

## A discriminating next test

Before new optimization, establish whether a candidate's native reader can
recover content from the actual fixed Chinese/English image prompts without
answer-aware cropping or unreadable resizing. A temporary source text readout
is a diagnostic only. Separately test whether the native image path can render
short exact strings when explicitly supplied the strings; that assisted control
is not an ILM answer. If either source capability fails, replacing a vocabulary
head cannot be claimed to fix it.

Only a subsequent trained image-question to answer-image path qualifies as the
student. Its changing conditions must come from prompt pixels, with fixed
structural controls compiled offline. Retain the native continuous generator;
remove runtime language lookup/output heads only after verifying the required
numeric controls and unchanged native computation. Do not route a teacher's
decoded answer to a font renderer. Evaluate unrepaired output on new strings,
held-out natural questions and two different questions about the same passage.
Report accuracy and allocated memory before any efficiency comparison.

A training-only differentiable recognition loss is a separate hypothesis for
the observed character omissions and repetitions. It requires a calibrated
recognizer and actual gradients through generated pixels, not Tesseract
backpropagation. It must not replace the independent output assessment or
restrict historical forms to a modern OCR dictionary. Neither hypothesis is
currently validated.
