# A stronger writing prior would still need an autonomous reading test

The released Qwen-Image-2.1 metadata describes a 7.115B image transformer
**and an additional 8.767B Qwen3-VL encoder**. Their declared BF16 payloads
sum to 29.583 GiB before the VAE, activations or adapters. The 7B description
therefore does not describe the complete deployed system. This is a source
audit, not a local execution, fit measurement or answer-quality result.

## Pinned source and cost

The [audit record](evidence/qwen_image21_source_audit.json) binds the official
model revision `d26bb61231c349cf6b7896fa83353113880e1ba3` and Diffusers revision
`84532a058702c74ae5d1dd6ceb05683fc5d5e768`. Only nine small model metadata,
README and licence files and five library source/documentation files were
retrieved. No weights were acquired and no model or optimizer was run.

| Published component declaration | Source value |
| --- | ---: |
| Image-transformer BF16 payload | 14,230,249,472 bytes |
| Image-transformer parameters inferred at two bytes each | 7,115,124,736 |
| Qwen3-VL encoder declared parameters | 8,767,123,696 |
| Qwen3-VL encoder BF16 payload | 17,534,247,392 bytes |
| Combined declared parameters, excluding VAE | 15,882,248,432 |
| Combined BF16 payload, excluding VAE | 29.583 GiB |

These are the released index declarations, not parameters enumerated from
loaded weights or independent whole-file checksum verification. An entirely
resident BF16 copy of both published components exceeds a 24 GiB card before
other memory. Quantization or sequential offload could change feasibility,
but this audit measures neither its memory nor its latency. The source is the
[transformer weight index](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/transformer/diffusion_pytorch_model.safetensors.index.json)
and [encoder weight index](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/text_encoder/model.safetensors.index.json).

## Information path

The pinned native pipeline supplies image pixels and prompt IDs to Qwen3-VL,
then passes continuous hidden states to the generator. Its prompt-embedding
method contains one encoder forward and no text-generation call. Reference
images also condition the VAE path. The pipeline defines sequential component
offload and condition-cache reuse. These interfaces make an image-conditioned
experiment conceivable; they do not show that it can infer the wording of an
answer. [Pinned pipeline source](https://github.com/huggingface/diffusers/blob/84532a058702c74ae5d1dd6ceb05683fc5d5e768/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py).

The same source explicitly preserves a pre-final-normalization hidden-state
boundary across Transformers versions. A future adapter must reproduce this
boundary, the image-pad mask and native preprocessing before evaluating a
new objective. Fixed-task compilation and removal of unused lexical modules
would require actual numerical replay. Neither is implemented or verified
for this candidate here. The source's image-to-image LoRA example offers
offload, checkpointing and quantized-generator options, but supplies no local
single-card learning result. [Pinned training example](https://github.com/huggingface/diffusers/blob/84532a058702c74ae5d1dd6ceb05683fc5d5e768/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py).

The release carries the Qwen Research License rather than an Apache licence.
Its terms must accompany any decision to reuse or distribute this checkpoint.
[Release licence](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE).

## Decision

No foundation switch follows from rendering examples alone. The active
DeepGen training and its registered complete assessment remain unchanged.
If a later candidate is tested, its minimum evidence must include autonomous
image-question answering with a generic fixed instruction, paired questions,
erasure controls and original answer pixels. Supplied-answer rendering must
be reported separately. All encoder/generator/VAE parameters and inherited
pretraining must count in any efficiency comparison. This audit supplies a
bounded source candidate, not an independent ILM or a demonstrated remedy.

```bash
CUDA_VISIBLE_DEVICES='' python scripts/audit_qwen_image21_source.py \
  --source artifacts/external/qwen_image_21_source_audit_v1 \
  --output artifacts/qwen_image_21_source_audit_v1
```

Use a fresh output directory. The record includes each pinned source URL,
byte length and SHA256; these small files can be reacquired from those URLs.
