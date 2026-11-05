# Literal writing transfer after full-corpus training

This diagnostic tests a basic image-language requirement: reproduce writing
requested in a new image prompt. It does not replace the original language
audit, decide access to sealed examples, or qualify a general ILM.

Use the 12 fixed cases in scripts/prepare_copy_transfer_v75.py: four Chinese
phrases, four English phrases and four six-digit strings. Cases form six
pairs, with different requested writing under the same instruction pattern.
The source strings serve only the offline renderer. Prompt and reference
images are stored separately; the neural actor receives only prompt pixels.
No teacher, OCR, character IDs or reference length enters generation.

Reuse the qualified complete-line renderer: Noto Sans CJK JP, 24 pixels,
32-pixel height, complete ink bounds and 64-pixel width padding. Fold prompts
with the existing physical layout algorithm. Keep all source marks without
cropping or changing font size. Confirm that none of these 12 exact prompt
rasters occurs in the full admitted training corpus. Only the current
training images and these new author-specified diagnostic cases are used in
preparation; original sealed examples remain untouched.

Evaluate only the completed 40,000-update full-corpus clean checkpoint,
after the original 384-call audit and independent CPU verification finish.
Use seed 20267580 plus pair index, common within each pair; retain the ordinary
128-block safety cap, 32 clean-endpoint updates and learned stopping at 0.5.
Re-encode actual emitted pixels as causal history. Preserve every raw answer.
Load reference pixels only after the corresponding autonomous call.

Report all 12 original-reference exact matches, foreground F1, stopping and
both-exact pair counts, by requested script and overall. Also score each
answer against its pair partner's reference. These are descriptive diagnostic
results, with no new pass threshold and no extra training/checkpoint selection.
An exact-prompt-disjoint copy case is not proof of novel-character generation,
unseen knowledge or the absence of foundation-pretraining contamination.

An independent CPU stage authenticates source/provenance, reconstructs input
layout, checks raw images and stopping events, and independently recomputes
physical scores. It does not replay the entire neural forward pass or every
feedback encoding. Keep this result separate from the original full-answer
qualification, whether it succeeds or fails.

Preparation is CPU-only and loads no neural model. Evaluation uses the shared
project GPU lock and physical GPU 0 after the main pipeline has exited:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libglib-2.0.so.0:/lib/x86_64-linux-gnu/libgobject-2.0.so.0:/lib/x86_64-linux-gnu/libgio-2.0.so.0 \
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python scripts/prepare_copy_transfer_v75.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/audit_copy_transfer_v75.py --stage evaluate
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python \
  scripts/audit_copy_transfer_v75.py --stage verify
```
