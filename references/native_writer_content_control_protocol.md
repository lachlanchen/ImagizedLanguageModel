# Supplied-content control of the native image writer

## Question and boundary

The completed image-only flow and shared-reader-query experiments produce
incorrect answers even on their training questions. This control asks whether
the pinned native writer produces correct writing when the intended answer
content is explicitly supplied through its original text-conditioning path.
It does not test question understanding or language acquisition. Text prompts,
the source tokenizer and its lexical input table are deliberately retained
only in this offline diagnostic. They are not an admitted ILM deployment.

The [released pipeline](https://huggingface.co/deepgenteam/DeepGen-1.0-diffusers)
supports a text prompt with a reference image. Use the already verified local
revision `85c2ed257b9406d8531965c68e140dc7fdb81382`, its exact native sampler,
and the existing source-version compatibility layer. No downloads, teacher
answering call, output OCR repair, font rendering of model predictions,
optimizer update or new adapter training is part of this control.

## Fixed cases and comparison

Before model loading, register exactly four cases from the completed
`native_query_fit_v1` preparation, in this order:

1. Its first seen Chinese question and supplied canonical answer.
2. Its second seen Chinese question and supplied canonical answer.
3. Its first seen English copying case and supplied reference string.
4. Its first excluded English copying case and supplied reference string.

Keep the complete original 1024-square input images. The four clean target
images are used for separate CPU OCR calibration and later review, never as
generator reference images or starting latents. This is reused development
data, not an independent generalization set. The copying split names describe
the earlier optimizer exposure, not exposure of a pretrained source model.

Supply each reference string verbatim in the same fixed instruction template:

```text
Replace the reference image with a clean white page. Write exactly the following text in clear black print, preserving every character, word and its order:
{reference_text}
Only this text should appear. Do not add labels, headings, quotation marks, explanations or decoration.
```

Compare `source` and `adapted` arms. Both use the original continuous reader
queries, native lexical/text conditioning and a rank-16 connector/writer
adapter graph. The source arm has the original zero-effect adapters; the
adapted arm loads the exact completed 1,536-update adapter export. The failed
128-update reader-query fit is not loaded. Retain and report all pretrained
components and the unused output head. These arms therefore differ only in
the registered connector/writer adapters, not in a learned vocabulary.

For each arm, first replay the first registered Chinese question at the
original fixed generic task, original sampling seed, fifty steps and guidance
one. Compare all reader forward-input tensors, decoded tensors and PNG pixels
with that arm's original guidance-one sampling diagnostic. Require bit-exact
agreement before its supplied-content draws. A failed replay is an execution
failure; retain it and stop, rather than interpret subsequent quality.

Then generate all four supplied-content pages at guidance one and all four
at guidance four, always fifty native steps and 1024 x 1024 output pixels.
The fixed seed for case index i is `20270501+i`, shared across both arms and
both guidance settings. Complete order is source replay, source guidance-one
cases, source guidance-four cases, adapted replay, adapted guidance-one cases,
adapted guidance-four cases. Total: sixteen supplied-content pages, two
execution replay pages, 900 denoiser calls and 1,300 denoiser batch elements.
No per-output prompt change, reroll, extra generation seed or best-of selection
is admitted. Save actual decoder and reader-input tensors for every draw.

## Assessment and interpretation

Calibrate all four original target images with the existing fixed Tesseract
binary, language files and NFC/whitespace exact rule before GPU admission.
Preserve case and punctuation. The actual model child must exit before a
separate CUDA-hidden CPU child scores every original output and authenticates
its saved decoded tensor against the PNG pixels. Recompute the two execution
replays independently from the saved original tensors. Inspect all eighteen
complete generated pages and the four original targets without contrast
repair, replacing writing, cropping model output or manually repairing scores.

Report exact counts out of four separately for each arm/guidance cell and
each language's count out of two. Report the complete raw OCR strings, clean
reference calibration, peak allocated VRAM, time and runtime parameters.
These small repeated cases are descriptive; no significance claim is made.
Four of four exact pages in a cell would establish only a supplied-content
writing control on these cases. A lower count leaves that writing control
unqualified. Neither outcome establishes autonomous image-question answering,
semantic learning, historical-script generality, novelty or efficiency at
matched language quality. Failure would not uniquely identify the writer:
prompt conditioning, source-image interference and source capabilities also
remain possible causes. Do not scale the unchanged failed training recipe
merely because these supplied-content outputs improve.

## Execution and preservation

Commit this protocol and its runner before preparation; bind all files and
all 42 predecessor code/data bindings, exact adapter export and original
sampling counterpart indices. Refuse any changed binding and preserve any
existing execution directory. Retain the original 24 experiment sources.

Use only physical GPU0 under `artifacts/ilm_gpu0.lock`, with at least 20 GiB
GPU memory free, 24 GiB available RAM and no more than 75% swap use. Preserve
foreign workloads and GPU1. The torch-free coordinator may wait at most one
hour, polling every thirty seconds; the single GPU child has a thirty-minute
wall-time bound and a 23 GiB peak-allocation bound. It must not automatically
restart a failed or partially completed attempt. On cancellation, stop only
its exact child process group and record terminal state. No GUI is required.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/run_native_writer_content_control.py \
  --stage prepare --output artifacts/native_writer_content_control_v1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_native_writer_content_control.py \
  --stage coordinate --output artifacts/native_writer_content_control_v1
```

The coordinator uses the verified existing LocalLLM image-generation Python
environment when launched from that environment. All stages retain original
failure evidence. General ILM success remains a separate unachieved goal.
