# Full-dimensional glyph denoising control

The completed V72 writer fails its language requirements. This control tests
one narrower prerequisite: whether a flow can repeatedly generate one known
physical writing block from independent Gaussian noise. Passing is not language
understanding, image-conditioned copying, or a usable ILM.

The V72 velocity head maps 1,536 noisy coordinates through width 768 and ends
with a 768-to-1,536 affine projection. Its velocity therefore lies in a fixed
subspace. The new flow preserves all coordinates with width 1,536 and an
identity-initialized noise projection. Its final projection remains learned.
This removes an architectural ceiling; learned weights and generated pixels
still require measurement.

## Fixed selection and comparison

Use the first CMRC record in the completed, authenticated V72 cache order, and
the first 32-by-64 block of its complete original answer. Retain the source
hash, cached-code hash and original pixels. Do not choose another image after
seeing its reconstruction. The frozen V34 codec must first reconstruct this
selected original block with at least 0.99 foreground F1.

Train narrow and full-dimensional flows sequentially from scratch. Both use
an identity initialization of the rectangular/square noisy-input projection,
zero input bias, the original three residual MLP blocks, and a zero-initialized
final projection. Both receive a constant zero 512-vector as conditioning.
The target image is represented by its two frozen 768-dimensional codes.
There is no prompt-reading experiment in this control.

Each arm receives 4,000 AdamW updates, batch 64, betas (0.9,0.95), weight decay
0.01 and clipping at one. Learning rate is 5e-4 times a 100-update linear warmup
and cosine factor decaying from one to 0.1 over the fixed budget. Train on
straight paths between independent standard Gaussian noise and the fixed
target code, with uniform time and squared velocity loss. Use BF16
contractions and FP32 parameters/optimizer, matching V72 arithmetic.

Model seed is 20267300. A separate training generator uses 20267301 in each
arm, giving common noise/time draws. Save model, optimizer and generator every
500 updates. Do not select intermediate checkpoints by generated quality.

## Evaluation and acceptance

After update 4,000, generate all 64 outputs using generator seed 20267302,
32 midpoint steps, per-vector LayerNorm, the frozen FP32 decoder and the
original zero threshold. Save initial noise, unnormalized endpoints,
normalized codes and every unmodified binary output. The preview grid only
rearranges the retained physical blocks.

The full-dimensional arm must meet all three image requirements:

- Mean foreground F1 against the original image is at least 0.995.
- Every seed has original-image foreground F1 of at least 0.99.
- At least 90% of seeds exactly reproduce the supplied-code render.

Also report original-image exact agreement and all narrow-arm failures.
Supplied-code render agreement is distinct from original-image agreement.
No failed pixels may be corrected, reranked or omitted. Independent CPU
verification must recompute physical scores, replay the fixed decoder from
retained predicted codes, and authenticate preview pixels. It does not replay
the entire GPU neural sampler.

This control's failure does not justify a full language run. Its success
establishes only a denoising prerequisite. Prompt-conditioned semantics, full
answer generation, generalization and the original language gates remain
separate requirements.

## Reproduction

Use the existing compatible Python environment from the repository root.
Physical GPU 0 must be idle; the script checks host memory and holds the
project's exclusive model lock. No inherited reader or teacher is loaded.

~~~sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/probe_glyph_denoising_v73.py --out NEW_CONTROL_DIRECTORY

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/probe_glyph_denoising_v73.py --out NEW_CONTROL_DIRECTORY --verify
~~~
