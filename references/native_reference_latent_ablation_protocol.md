# Isolating source-image VAE reference conditioning

## Question and scope

The completed supplied-content writing control leaves reliable writing
unqualified. This experiment tests one specific interference hypothesis:
does the source-image VAE reference input impede generation of the supplied
answer while the model continues to read the same source image?

Use the same pinned DeepGen source, original reader queries, source/adapted
connector and writer adapters, complete source images, supplied-content text
prompts, native sampler, guidance four and seeds as the earlier
[control](native_writer_content_control_protocol.md). Keep both the visual
reader and the supplied text-conditioning path active. Remove only
`cond_hidden_states` at the image denoiser's forward boundary. Do not replace
the input page with a blank image, remove image features from the reader,
change prompts, resize or crop inputs, or supply target pixels.

The [pinned native implementation](https://huggingface.co/deepgenteam/DeepGen-1.0-diffusers/blob/85c2ed257b9406d8531965c68e140dc7fdb81382/deepgen_pipeline.py)
accepts `cond_hidden_states=None`; it then omits the appended source-image
latent patches while retaining the generated image's latent patches and
semantic conditions. Use a recorded PyTorch forward pre-hook to set exactly
that keyword to `None`. A subsequent observing hook must see the actual
changed argument. Both hooks leave every other argument object unchanged.
No source weight or published pipeline file is edited.

For reader condition c(x,a), source VAE reference r(x), and evolving target
latent z, the intervention compares v(z,t,c(x,a),r(x)) with
v(z,t,c(x,a),None). The supplied answer text a is deliberate diagnostic
supervision. Neither branch is an independent image-language deployment.
No optimizer update, new adapter fitting, output rendering or repair occurs.

## Fixed complete comparison

Reuse the exact four registered Chinese/English cases and their prompt bytes
from `native_writer_content_control_v1/cases.json`, in the same order. Reuse
its checked four-target reference OCR calibration under an unchanged engine.
Bind the complete prior preparation, execution, output index, calibration,
assessment, CPU audit, actual-review receipt and model adapter export.

Use the original source and 1,536-update adapted arms. The failed 128-update
reader-query fit is not loaded. For each arm:

1. With the intervention disabled, replay case zero's supplied-content output
   at guidance four and the original seed. Require bit-exact reader inputs,
   decoder tensor and PNG pixels against the original control. Retain the
   actual first denoiser inputs and source reference latents.
2. Enable reference removal. Generate all four cases at their unchanged seeds,
   fifty steps and 1024-square output. Compare every case's reader-input
   tensors with its original full-reference counterpart. All must be bit-exact.
3. For case zero, also require first-step generated-noise, timestep, semantic
   sequence and pooled-condition tensors to equal that arm's execution replay.
   Thus the reference input alone differs at the first denoiser call.

Complete order: source replay, four source removals, adapted replay, four
adapted removals. Total: ten new draws, 500 denoiser calls, 1,000 denoiser
batch elements. Eight original full-reference counterparts are reused without
regeneration. For all 400 ablated calls, the reference must exist before the
intervening hook and be absent at the subsequent observer. For both replays,
all 100 calls must preserve the reference. Retain complete original PNGs,
actual decoder/reader tensors, first denoiser inputs, actual reference-latent
tensors and per-call observations. No reroll or best-of selection is allowed.

## Assessment and interpretation

Only after the GPU child exits may a separate CUDA-hidden CPU child verify
all ten actual PNG/decoder bindings, both complete replays, all eight reader
input comparisons, both first-step isolation comparisons and all 500 hook
observations. Use the same offline OCR binary, language files and exact rule:
NFC and whitespace normalization, with case and punctuation preserved.
Recompute original full-reference counts from their unchanged raw OCR, and
report paired full-reference/reference-removed counts out of four and out of
two per language. Inspect all ten complete new pages and retain all failures.
Original counterpart and reference pages were already actually inspected;
require their unchanged hashes rather than claim a new blind assessment.

Four exact ablated pages in an arm would qualify only this small supplied-
content writing control. Lower counts leave it unqualified. A gain is not
proof of language understanding, language acquisition, historical-script
generality, general performance or efficiency at matched quality. A failed
control does not establish that all reference-free architectures or other
source prompts fail. The observed intervention changes reference conditioning
and the associated image-patch attention geometry together; do not attribute
an effect uniquely to a particular attention mechanism.

## Resource and execution rules

Commit the complete runner and protocol before preparation, register exact
cases and counterpart hashes before generation, and preserve all 26 previous
core source files. No new model acquisition or duplicate weights are allowed.
Use GPU0 only and the existing project lock. Admission requires at least
20 GiB free GPU0 memory, 24 GiB available RAM and swap use at most 75 percent.
The torch-free coordinator may wait one hour, polling every thirty seconds.
Allow one thirty-minute GPU execution, at most 23 GiB peak allocation, followed
by one separate CPU verification/scoring child. On failure retain the attempt
and all completed draws; never automatically restart. Preserve foreign work
and GPU1. No GUI or publication build runs concurrently with the model.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/run_native_reference_latent_ablation.py \
  --stage prepare --output artifacts/native_reference_latent_ablation_v1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_native_reference_latent_ablation.py \
  --stage coordinate --output artifacts/native_reference_latent_ablation_v1
```

Run with the existing verified LocalLLM image-generation Python environment.
The full ILM goal remains meaningful image-prompt to image-answer language
generation, independently of the outcome of this component control.
