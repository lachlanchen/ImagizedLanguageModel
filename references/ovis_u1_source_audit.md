# A compact jointly trained source candidate

Ovis-U1 provides a smaller complete source to examine for image-conditioned
writing. Its technical report lists about 3.644B parameters across the language
body, visual encoder, adapter, image decoder, VAE and refiner. The 1.7B language
backbone and approximately 1B image decoder are parts of that total. Its joint
training includes understanding, generation and editing; those published task
results do not establish autonomous image-question/image-answer language
generation. [Author technical report, Table 1](https://arxiv.org/html/2506.23044v1#S2.T1).

## Actual metadata inspection

The original AIDC-AI model link redirects to the author's current
`ATH-MaaS/Ovis-U1-3B` repository. The [audit](evidence/ovis_u1_source_audit.json)
pins revision `cbc005ddff7376a20bc98a89136d088e0f7e1623`, authenticates twelve
small source/configuration/licence files and checks all three published
weight-shard declarations. No weight shard is downloaded or model imported.

| Released archive declaration | Value |
| --- | ---: |
| BF16 floating elements | 2,164,776,960 |
| FP32 floating elements | 1,479,431,407 |
| Total declared floating elements | 3,644,208,367 |
| Tensor payload | 10,247,279,548 bytes (9.544 GiB) |
| Weight shards | 3 |
| VAE tensors included in those shards | Yes |

The mixed-dtype payload agrees exactly with the published index's byte count.
These are archive declarations, not parameters enumerated from a loaded
runtime or measured GPU allocation. Aliases, loading dtype, activations and
temporary buffers require an actual memory check. The model's native generation
quality and one-card training feasibility remain unmeasured here.
[Pinned configuration](https://huggingface.co/ATH-MaaS/Ovis-U1-3B/blob/cbc005ddff7376a20bc98a89136d088e0f7e1623/config.json),
[weight index](https://huggingface.co/ATH-MaaS/Ovis-U1-3B/blob/cbc005ddff7376a20bc98a89136d088e0f7e1623/model.safetensors.index.json).

## Relevant image path and remaining boundary work

The released `generate_condition` method obtains two final language-model
hidden layers from merged image/prompt embeddings. `generate_img` passes
that condition to the native image decoder. These methods contain no
autoregressive text-generation call; the separate text API does. This offers
an image-conditioning route without supplying a generated textual answer
to a renderer. It does not show that the route can infer the correct answer.
[Pinned model source](https://huggingface.co/ATH-MaaS/Ovis-U1-3B/blob/cbc005ddff7376a20bc98a89136d088e0f7e1623/modeling_ovis_u1.py).

The default visual adapter uses softmax over 65,536 learned visual prototypes.
Its floating-input embedding path performs a matrix multiplication, while an
integer lookup branch also exists. This is explicit learned visual vocabulary
machinery and must not be advertised as vocabulary-free. The native system
also retains a text tokenizer and lexical modules. A fixed-task image-only
runtime would require explicit compilation, narrowly targeted removal and
actual numerical replay; none is implemented for this candidate here.

Source compatibility also remains open: the release specifies Transformers
4.51.3, while the current experiment uses a different verified environment.
No package or running-model environment has been changed. Any later execution
must preserve native preprocessing, layer-normalization boundaries, latent
scaling and image-guidance conventions before drawing a quality conclusion.

## Research decision

This is a source candidate, not a verified replacement for the active model.
The registered complete DeepGen assessment and queued conditional-field probe
continue unchanged. A candidate preflight would need actual complete-input
copying and paired-question image outputs, source-equivalent execution and
measured full-system cost. Inherited pretraining remains explicit. A rendering
benchmark, small decoder or successful load cannot qualify the ILM objective.

```bash
CUDA_VISIBLE_DEVICES='' python scripts/audit_ovis_u1_source.py \
  --source artifacts/external/ovis_u1_source_audit_v1 \
  --output artifacts/ovis_u1_source_audit_v1
```

Use a fresh output directory. Each small file's exact pinned URL, length and
SHA256, plus the declared whole-weight SHA256s, are retained in the record.
Whole-weight verification, neural execution and local answer-quality results
are absent.
