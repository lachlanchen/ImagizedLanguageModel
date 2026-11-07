# Native visual reading and continuous-writing research

The [source diagnostic](../references/native_unified_source_result.md) fails on
the actual glyph prompts despite retained text-language capability. The
[learning direction](../references/native_visual_language_learning_direction.md)
first repairs visual access through a frozen auxiliary language head, then
requires direct continuous image-answer learning. The current implementation
is the source diagnostic and visual-access learner. It does not yet provide
the required independent image-answer ILM.

## Reuse pinned assets and isolate dependencies

From the repository root, use the existing shared image-generation environment
for acquisition and setup. Only the native checkpoint, VAE and small component
configs/tokenizer files are acquired. Model and source hashes are locked.

```bash
../LocalLLM/.local/image-generation/venv/bin/python scripts/acquire_native_unified_source.py --stage acquire
../LocalLLM/.local/image-generation/venv/bin/python scripts/acquire_native_unified_source.py --stage runtime
export ILM_NATIVE_PYTHON="$PWD/artifacts/external/native_unified_source/runtime/bin/python"
env CUDA_VISIBLE_DEVICES= "$ILM_NATIVE_PYTHON" scripts/acquire_native_unified_source.py --stage verify
```

The isolated runtime pins Transformers/Diffusers dependencies and uses a `.pth`
reference to the existing Torch environment. It preserves the venv interpreter
path and does not install over the shared environment. Official source is
Apache-2.0 and remains hash-verified outside Git. Two tuple-helper imports use
torch's equivalent integer/tuple helper; neural forward functions are unchanged.
Constructor-only SigLIP hydration uses config rather than redundant weights,
followed by strict checkpoint loading and checked native weight tying.

## Source diagnostics

The existing prepared image data are required. Fresh preparation is described
in [paired-image learning](paired-image-learning.md); it needs the public CMRC
fixture and installed Noto Sans CJK font, without an earlier trained checkpoint.
Supply `--manifest` and `--copy-manifest` if using different prepared paths.

```bash
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/probe_native_unified_source.py --stage prepare --output artifacts/native_unified_probe/prepare_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/probe_native_unified_source.py --stage reading --inputs artifacts/native_unified_probe/prepare_v1/inputs.json --output artifacts/native_unified_probe/reading_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/probe_native_unified_source.py --stage writing --inputs artifacts/native_unified_probe/prepare_v1/assisted_writing.json --output artifacts/native_unified_probe/writing_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/probe_native_unified_text_control.py --output artifacts/native_unified_probe/text_control_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/check_native_reading_cache.py --output artifacts/native_unified_probe/cache_replay_v1
```

Run GPU stages sequentially. They hold the project GPU lock and enforce the
shared-memory limit. Output directories are protected against replacement.
The writing stage explicitly receives desired strings and is an assisted
renderer control. The text control explicitly receives original text. Neither
is an image-only student. Fresh reading receives an input-only index; all
expected strings and offline OCR remain separate.

## Visual-access optimization

The implemented learner adds gated rank-16 residuals only to the native semantic
visual encoder and fusion linear layers. Base weights, language body/head and
generator remain frozen. The gate stays active through checkpointed backward
recomputation and can be disabled to recover the base visual path. FP32 adapter
masters and optimizer state use BF16 native computation. Native last-position
logits avoid projecting every source position over the language vocabulary.

```bash
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_access.py --stage cache --output artifacts/native_visual_access/cache_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_access.py --stage preflight --output artifacts/native_visual_access/preflight_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_access.py --stage evaluate --output artifacts/native_visual_access/before_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_access.py --stage train --updates 4096 --output artifacts/native_visual_access/learning_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_access.py --stage evaluate --checkpoint artifacts/native_visual_access/learning_v1/checkpoint.pt --output artifacts/native_visual_access/after_v1
```

The cache contains all 448 input-image latents: 384 training records and 64
validation records. Only the declared training IDs enter optimization. Native
FP32 VAE samples preserve full input geometry; the original twelve diagnostic
latent tensors match exactly. The two-step preflight uses a long training copy
and the largest training QA prefix, executes real optimizer updates and checks
finite/nonzero gradients and unchanged disabled-gate prefix. It establishes
execution at 6.899 GiB peak, not language quality.

Training alternates shuffled cyclic copy/QA lists, saves every 128 updates and
records each chosen ID, loss and gradient coverage. Checkpoints contain learned
residuals, optimizer/RNG state and source/data/code bindings. Resume unchanged
code into a new output directory:

```bash
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_access.py --stage train --updates 4096 --resume artifacts/native_visual_access/learning_v1/checkpoint.pt --output artifacts/native_visual_access/learning_resumed
```

Resume needs the same pinned source and cache manifest, without an ancestor
checkpoint chain. Documentation-only Git commits do not invalidate it. Do not
change bound implementation files during a live run. Check status and original
logs before starting another job; an observation timeout is not termination.
In this workstation run, the private sequential controller starts the fixed
after-learning readout after the 4,096-update process exits successfully.

Auxiliary supervision explicitly uses Unicode/token labels and correct preceding
answer labels during training. Autonomous evaluation reads fresh input images,
numeric adapter state and source weights; it does not load training cache or
reference answers. Its temporary native text head remains a diagnostic. Correct
held-out image answers, question interventions, reliable Chinese writing and
eventual vocabulary-free image deployment are still required for the full goal.
