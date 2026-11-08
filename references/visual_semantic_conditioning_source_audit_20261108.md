# Does the image generator actually receive visual language features?

The current writer supplies VAE reference features and one fixed generic
condition to a pretrained generator. The initially inspected checkpoint had
not optimized natural QA; a subsequent registered full-page QA learning run
is now in progress. Its interface does not by itself provide an already
trained visual-language reading path. Better copying cannot establish that this body understands a
paragraph or its question. This audit checks a possible warm start for that
specific gap; it does not replace the running controlled experiment.

## OmniGen2: distinguish the generation path from the understanding model

At repository revision `18e6f9d5271b517fcb32e999f10df943ae9b8f20`, the
[generation pipeline](https://github.com/VectorSpaceLab/OmniGen2/blob/18e6f9d5271b517fcb32e999f10df943ae9b8f20/omnigen2/pipelines/omnigen2/pipeline_omnigen2.py)
encodes its text prompt through `_get_qwen2_prompt_embeds`. That function
tokenizes text and calls the MLLM without image pixels. Reference images
instead enter `prepare_image` and its VAE branch. Therefore this particular
image-generation path does not establish that Qwen's visual reader encodes
the reference page. A separate understanding interface must not be assumed
to feed the same generator. Simply swapping to this pipeline would not test
the proposed semantic-input correction.

This is a call-path observation about the inspected implementation, not a
claim that the whole OmniGen2 system lacks visual understanding. The
[paper](https://arxiv.org/abs/2506.18871v4) describes distinct text and image
decoding pathways. No OmniGen2 weights or local model calls were used here.

## DeepGen: an explicit visual-semantic path is present

The [DeepGen paper](https://arxiv.org/abs/2602.12205v2) proposes a visual-language
body connected to a diffusion generator using hierarchical features. The
authors' benchmark results are external claims, not ILM measurements.

At revision `fbaaaf4ae1c79450d6a34b30beb4f59840568c69`, its
[SCB implementation](https://github.com/deepgenteam/deepgen/blob/fbaaaf4ae1c79450d6a34b30beb4f59840568c69/src/models/sd3_kontext/qwen2_5_vl_sd3_hf_dynamic_fusion.py)
has a concrete image-to-vision-to-language-body-to-generator path.
`get_semantic_features_dynamic` invokes the vision tower; image embeddings
enter `prepare_forward_input` alongside learned continuous queries.
`generate` calls the language body forward, fuses selected hidden layers and
passes continuous conditioning to the diffusion pipeline. It does not sample
answer words in that function. A separate VAE branch preserves appearance.
The dynamic visual path scales spatial dimensions by 28/32 rather than
necessarily forcing every input to 448 square. These facts make it a candidate
for an image-question/image-answer baseline, without proving it can answer
our questions or write Chinese reliably.

The released path still tokenizes a text template and uses a vocabulary
embedding table. It is not yet the strict deployed ILM interface. A possible
adaptation is to compile only the fixed task/template embeddings and positions,
insert image features and continuous queries, and remove unused lexical input
lookup and output heads. This is an implementation hypothesis. Native forward
equality must be demonstrated before claiming removal preserves behavior.
The retained language-body pretraining must remain counted and disclosed.

## Published packaging requires verification

The [self-contained release](https://huggingface.co/deepgenteam/DeepGen-1.0-diffusers)
is pinned at `85c2ed257b9406d8531965c68e140dc7fdb81382`. Its downloaded Python
source has SHA256 `41e0f263d9ad55940426a4b2092688a74711395499c12a71ee69dc378ec9d3a9`;
the initial static inspection downloaded no model weights. Inspection finds
three material details:

- The wrapper resizes the input image to the **output** width and height.
  Using a 512×256 answer canvas would therefore squash a 1024-square book
  page. A full-page experiment needs a separately validated input geometry.
- VAE reference encoding samples its posterior without forwarding the supplied
  sampling generator. A replay must bind the global RNG as well as the denoiser
  generator; silently changing posterior sampling to its mode is a different
  experiment.
- `_load_extras` prefers the package's local `vlm/` folder. This pinned
  release actually includes that configuration and two complete VLM weight
  shards. The native package therefore resolves those files before its external
  Qwen fallback. Substituting base Qwen would be a different baseline. The
  different whole-file hashes do not by themselves prove different tensor
  values, since serialization and sharding can also change.

The [complete bounded-header audit](evidence/visual_conditioning_package_headers.json)
corrects the earlier incomplete 6,224,286,912 estimate. The hub's summary
2,469,663,936 count describes only the transformer file; it omits the
connector, VAE and bundled VLM. Every weight file's header is now checked:

| Packaged component | Stored tensor elements | Storage dtype |
| --- | ---: | --- |
| Transformer | 2,469,663,936 | BF16 |
| Connector | 432,449,532 | FP32 |
| VAE | 83,819,683 | BF16 |
| Bundled VLM | 3,754,622,976 | BF16 |
| Total | 6,740,556,127 | Mixed |

These five files total 14,346,247,966 bytes, including their headers. Stored
state includes a 226,492,416-element transformer positional buffer, so this
is not automatically a count of learned model parameters. Meta construction
of all four real component configurations also exposes a VLM head-sharing
distinction: its constructor initially creates an untied head absent from
the stored shard index. The [explicit meta weight-tying check](evidence/visual_conditioning_package_meta_ties.json)
leaves that prototype head untied under the current API. The constructor
therefore counts 6,825,228,639 parameters before any actual loading or unused
component removal. This is not a loaded-runtime count. Loader-time handling
must be checked; the observation does not prove that published weights are
corrupt or that the head is used by image generation.

The audit reads bounded header prefixes and configuration files only;
no tensor payload or weights are loaded. Entire weight-file checksums are
recorded from hub metadata, not independently verified by these partial reads.
`scripts/audit_visual_conditioning_package.py` reproduces the file accounting
and optional CPU meta-structure checks. No memory fit, native numerical
replay or language-quality result follows from these observations.

There is also a checkpoint question in the native code: its `state_dict`
method excludes every `lmm.` key, while the
[SFT configuration](https://github.com/deepgenteam/deepgen/blob/fbaaaf4ae1c79450d6a34b30beb4f59840568c69/configs/finetune/deepgen_joint_sft_scb.py)
requests VLM LoRA. This observation does not establish what all published
checkpoints contain. Before relying on a particular release, inspect its
actual keys and establish native replay with its resolved VLM/adapters.

## Decision

The physical-loss comparison and full-page probes are complete. Finish the
registered native QA learning run and its complete assessment. A separate
`acquire_visual_semantic_source.py` helper prepares the pinned complete release
in the project cache, verifying whole-file checksums and reusing matching
shared files. Source acquisition is separate from model loading or adoption.
DeepGen merits a bounded source-consistent baseline because the inspected
path carries image features through the visual-language body into generation. Do not assume it is a solution from its name,
benchmarks or nominal parameter count. No local memory fit, training speed,
Chinese answer quality or strict vocabulary-free deployment has been proved.

Any subsequent model must still receive the complete question as pixels,
generate unrepaired answer pixels from independent noise, and pass unseen
paired-question tests with controls. An external pretrained reader becoming
part of a self-contained model is a warm start, not independent foundation
training or a new physics result. No active training source, weight, schedule
or criterion was changed during this audit.

## Executed CPU compatibility check

The [code-only probe](evidence/visual_conditioning_compatibility.json) now
checks the pinned release in the actual image-generation Python: PyTorch
2.11.0, Diffusers 0.39.0 and Transformers 5.14.1. The unmodified source first
fails because `einops` is absent. Importing the checksum-verified einops 0.8.1
wheel from an isolated path lets the same source module import; no package
is installed into the training environment.

Constructing base Qwen from its pinned configuration on the meta device gives
3,754,622,976 architectural parameters, all without allocated parameter data.
Its body has `model.language_model` and `model.visual`, but the top-level
`language_model` and `visual` attributes used by the published wrapper are
absent. `model.get_rope_index` is available. Thus successful module import is
insufficient for this wrapper to run unchanged with the inspected API.
This does not assess the authors' original environment or establish that a
compatibility change preserves numerical behavior.

Reproduce with `scripts/probe_visual_conditioning_compatibility.py` in the
existing model Python, with CUDA hidden. The probe downloads only a small
pinned configuration and optional dependency wheel, loads no weights, performs
no forward pass and initializes no CUDA context. It leaves the active QA run
and its environment unchanged. Native numerical replay, full-page geometry,
actual memory fit and answer quality remain outstanding before adopting this
warm start.

## Fixed-task compilation

The [fixed visual task compiler](compiled_visual_task_condition.md) separates
offline task-template compilation from live image-feature assembly. Actual
layout checks at 1024 × 1024 and 512 × 256 use the released tokenizer and
the checkpoint's source-version position algorithm. The installed 5.14.1
`get_rope_index` additionally requires `mm_token_type_ids`, absent from the
released call, and initializes masked positions differently from 4.56.1.
Compilation executes a checksum-pinned copy of the original 4.56.1 method;
it does not install another Transformers version or silently substitute a
different position algorithm.

The deployed assembly component contains fixed vectors, masks and positions,
and accepts live continuous visual features and queries. A complete generation
model still needs native numerical replay, explicit removal of unused lexical
components and assessment on actual pages. Successful compilation does not
demonstrate any of those model-level outcomes.
