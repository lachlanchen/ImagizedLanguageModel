# Native source language survives, but visual reading fails this cohort

The completed Show-o2 HQ source diagnostic gives **0/12 canonical exact answers
from image prompts**. All eight copying-prompt transcriptions are also wrong.
The same native body gives **5/12 canonical exact answers from original text**:
4/8 copies and 1/4 natural questions. All eight requested copy contents appear
in the text responses, with unwanted quotes on the four Chinese copies. Three
natural answers differ from their reference in wording or specificity; exact
match is not a complete semantic assessment. Every response remains retained.

The [complete export](evidence/native_unified_source_result/result.json)
contains all twelve original input/reference images, twelve supplied-input VAE
reconstructions, eight assisted image-generation outputs and the original
reports/readouts. CPU export verifies all twenty generated PNGs against retained
decoder tensors. No OCR repair or font substitution is used. The source helper
retains a language vocabulary; these are initialization diagnostics, not an ILM
score or successful image-question-to-image-answer deployment.

## Separating the components

The input VAE reconstructions remain readable on inspection. From those image
features, the native source reader invents unrelated strings, including altered
English fragments and Chinese phrases absent from the page. All four natural
questions produce the same repetitive numeric response and reach the 96-decision
limit. The source-text control uses the exact text rendered into these same
images; all twelve full-text SHA256 values match the preparation metadata.
This demonstrates remaining text-language access, while visual access is
unreliable in this composition. It does not uniquely distinguish source
checkpoint specialization, visual features, image geometry or conditioning.

The source was separately given the eight desired strings explicitly for
image generation. Offline OCR gives **0/8 strict exact strings**, against
**10/12** on the clean reference pages. Some English writing is readable, with
capitalization or recognition differences; other outputs omit words or alter
letters. The four Chinese outputs are blank or visibly incorrect. These
assisted controls cannot count as image-only answers. Different fonts, page
photography and OCR uncertainty remain visible in the raw images.

| Retained invocation | Seconds | Peak allocated GPU memory (GiB) |
| --- | ---: | ---: |
| Twelve images, twenty FP32 text-helper readouts | 286.977 | 19.055 |
| Eight BF16 text-supplied image controls | 78.200 | 7.658 |
| Twelve original-text controls, FP32 | 15.552 | 11.304 |

These are single invocations on one RTX 4090 D, excluding asset acquisition
and external pretraining. No optimizer updates occur. The complete native model
contains 2,830,777,056 parameters; its language body contains 1,543,304,192, and
the VAE adds 126,892,531. The source/model are NUS Show Lab, Apache-2.0. The
[asset lock](native_unified_assets.lock.json) pins the source commit, checkpoint,
VAE and configuration/tokenizer files. Redundant Qwen/SigLIP weight downloads
are avoided; retained weights load strictly from the complete checkpoint.

## Numerical execution and next learning experiment

An incremental KV and last-position-logit path reproduces **all 512 decisions
in all twenty original FP32 readouts**. The matched decoder-call sum falls
from 268.322 to 18.061 seconds. Cached replay uses the same retained input
latents and excludes VAE encoding. This speeds the same incorrect answers;
it is not useful language efficiency or a quality improvement. The source
calls and replay remain separate complete observations.

The [visual-access learning protocol](native_visual_access_learning_protocol.md)
therefore tests supervision of the visual frontend through the frozen native
language head. Its [real two-update preflight](evidence/native_visual_access_preflight.json) reaches every one of 158 up/down
residual pairs, with 8,489,984 trainable parameters and 6.899 GiB allocated peak.
The zero-residual prefix matches, and disabling the gate restores the original
prefix exactly after updates. The 4,096-update run on all 384 training examples
has started; 64 validation examples are excluded. There is no trained reading
or image-answer result yet. The [complete direction](native_visual_language_learning_direction.md)
keeps direct continuous image writing and removal of the runtime language head
as required subsequent work.

Source image, writing and original-text protocols were fixed separately before
their respective calls. Input images are never centre-cropped or resized;
copy strips preserve all source ink, and QA pages remain complete. The source
position interpolation handles actual geometry, and generation hydrates its
nonpersistent index buffer to the 512-square grid. The isolated runtime uses
Torch 2.11.0, Transformers 4.47.0 and Diffusers 0.31.0, reusing the shared Torch
installation. Implementation, provenance, commands and limitations are in the
[execution guide](../docs/native-visual-access.md).

The four CMRC passage images and references retain the prepared fixture's
CC-BY-SA-4.0 attribution: CMRC 2018, Cui et al., EMNLP-IJCNLP 2019. Other copy
pages contain locally prepared synthetic strings. These reused exploratory
cases do not constitute sealed evaluation or a benchmark replication of the
source authors' published model results.
