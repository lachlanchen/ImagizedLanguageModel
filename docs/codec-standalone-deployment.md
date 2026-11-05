# Native deployment of the complete codec-language actor

The standalone implementation reconstructs the same image reader, contextual
body, adapters and writing codec using native tensor shards. It imports no
GGUF reader, shared-weight loader, data cache, tokenizer or teacher module.
Installed PyTorch, Transformers, NumPy and Pillow remain required; their
versions and the Python major/minor version are recorded and checked.

The [CPU preflight](../references/evidence/codec_standalone_preflight.json)
passes eight checks. It matches all 748 inherited tensor shapes and storage
dtypes against the authenticated archive, and partitions the complete 1,053
state tensors into inherited weights, fixed codec and trainable adaptation.
The real meta-device architecture contains 7,539,239,282 parameters, including
9,351,937 trainable parameters. Its calculated native weight storage is
16,264,805,832 bytes, or 15.148 GiB. This is a storage calculation, not measured
runtime GPU memory or proof that a small model has learned language.

The preflight also streams the actual frozen codec into temporary shards and
loads it in a fresh Python process from an empty working directory. Its
continuous outputs for two existing supplied image codes match exactly,
including the recorded binary images. Python-audited source-file opens,
network operations and subprocess creation are rejected. Missing, corrupted,
unlisted and incomplete state inventories fail validation. Temporary codec
weight copies are removed after the check.

This takes 20.610 seconds excluding interpreter startup, allocates no CUDA
memory and makes no complete-context forward call. It does not verify full
actor isolation, new prompt-to-answer behavior or final trained weights. The
V68 trainer and its fixed autonomous audit are still running/waiting; no full
weight bundle has been created.

## Export after the original language gates pass

The exporter requires the completed 10,000-update run, matching independent
384-generation audit receipt and the original bounded Chinese development
gates. A failed language experiment is retained rather than exported as a
qualified deployment. GPU 0 must be idle and shared host memory limits met.

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python scripts/export_codec_language_v68.py \
  --out artifacts/codec_language_v68_20261105/native_bundle
```

Export loads the source-authenticated final actor once, checks it against the
standalone meta architecture and copies one parameter at a time to CPU shards.
The nominal shard target is 256 MiB; an individual larger tensor would remain
unsplit. Tensor names, shapes, dtypes and physical-byte digests are recorded,
alongside complete file/source hashes and external pretraining provenance.
Optimizer state, training examples, token vocabularies and font assets are
excluded. Retain the installed environment instead of copying it into the
bundle. A completed or partial export directory cannot be overwritten.

## Verify actual independent inference

After export exits and releases GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python scripts/verify_codec_language_bundle.py \
  --bundle artifacts/codec_language_v68_20261105/native_bundle \
  --out artifacts/codec_language_v68_20261105/native_bundle_replay
```

This verifier selects the first fixed normal-case prompt from the completed
audit. The child receives its image alone, the same 128-block generation cap
and the standalone weights. It runs Python with isolated import mode, an empty
working directory, empty model caches and offline settings. A Python audit
hook denies file opens in the source/data and shared-model roots except the
explicit bundle, prompt and output paths. It rejects Python network and
subprocess operations after the initial GPU resource query. This is not an
operating-system sandbox for arbitrary native code.

All raw blocks, learned stop probabilities, termination, prompt pixels and
native PNG bytes must match the original autonomous answer. The proof is
written outside the immutable bundle. A manifest alone does not establish
isolation; this full-model replay has not yet run.

## Use the exported image interface

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 \
artifacts/external/mixar_env_py312/bin/python \
  artifacts/codec_language_v68_20261105/native_bundle/run.py \
  --image prompt.png --out artifacts/my_standalone_answer
```

The bundle loads its exact native weights directly, reconstructs the two
nonpersistent rotary buffers and refuses any remaining meta state. It uses
the [same explicit page/strip image boundary](codec-image-inference.md) and
preserves raw output and learned stopping. The model needs the observed prompt
pixels and its own generated history, with no source archive or model service.

These commands are implemented but await a qualified final actor. A bounded
Chinese development pass and exact independent replay would still leave broad
English evaluation, historical/unencoded writing and the general ILM goal
open. They do not establish foundation training on one GPU or frontier parity.
