# Contextual adaptation for image-language generation

The current candidate trains the final four attention layers and the glyph
writer together. Its purpose is to repair the mismatch exposed by the
[native-step comparison](../references/glyph_feedback_ceiling_result.md).
It is not yet a demonstrated usable language model. The main cache, fit,
qualification and complete-answer assessment have separate receipts.

The authenticated body has 36 layers. Let the full causal sequence contain the
prompt image features followed by the reference glyph images' feedback vectors.
With vision, E and layers 0–31 fixed, define

\[
H_{32}=F_{0:31}(X),\qquad
h=\operatorname{Norm}(F_{32:35,\theta}(H_{32})).
\]

The cached tensor retains every prefix position, its coordinates and the
reference-image boundary. The changing last four layers still attend over the
complete causal sequence. Since the prefix is independent of the trainable
parameters, differentiating through this cached factorization gives the same
restricted-parameter objective as recomputing the frozen prefix. It does not
detach the outputs of layers that are being trained. Autoregressive inference
uses the full body and its new adapter weights.

Training minimizes balanced pixel BCE through the frozen whole-glyph decoder,
plus balanced stop/continue loss. Source-anchor regression is omitted. Only
688,128 contextual adapter parameters and 17,848,833 writer/stopping parameters
train. The lower body, image encoder, glyph decoder and affine feedback
normalization stay fixed. Three question-sequence updates alternate with one
old-state copying rehearsal update; the rehearsal regularizes the writer and
does not prove that the adapted body still copies correctly. Qualification
therefore runs real held copy images through the actual changed body.

The corrected preflight reproduces full-body output exactly on the checked
sequence, with zero maximum absolute difference. A changed future glyph leaves
earlier predictor states unchanged. Two real updates change 294,912 adapter-up
coordinates with nonzero gradients and no frozen-parameter changes, at 3.208 GiB
peak GPU allocation. These are implementation and resource checks, not language
accuracy or evidence that two updates suffice. The main fit uses 3,176 supported
training questions and 18,000 fixed updates. Thirty-five source cases are
recorded as missing from the older image cache or outside primitive coverage.
The inherited image cache contains occasional missing-form placeholders and
source omissions; the input review preserves those limitations.

The running recipe reuses the existing verified shared source weights. It caches
FP32 frozen-prefix states on disk, then loads only the four final layers for
fitting. The full body remains necessary at inference. Cache time, disk use and
inherited pretraining belong in the compute account; a small fitting footprint
is not a smaller foundation or a matched-quality speed advantage over text LLMs.

From the repository root, with the existing Python environment and authenticated
shared assets available:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/train_glyph_tail_adapter.py \
  --stage prepare --output artifacts/my_glyph_tail_run
# Inspect and record all 32 reserved input images before the main fit.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. \
python scripts/train_glyph_tail_adapter.py --stage cache-preflight \
  --output artifacts/my_glyph_tail_run
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. \
python scripts/train_glyph_tail_adapter.py --stage fit-preflight \
  --output artifacts/my_glyph_tail_run
PYTHONPATH=. python scripts/run_glyph_tail_pipeline.py --run artifacts/my_glyph_tail_run
```

The coordinator runs cache, fit and qualification serially under the GPU mutex,
then stops for actual image review. It preserves failed stages and nonzero exits
for inspection. It never invents a visual-review receipt or starts a second
model job. The current corrected run is `artifacts/glyph_tail_adaptation_v2`;
its preparation reuses the identical input images from v1, with the initial
layer-count error and first preflight preserved explicitly.

After qualification review, `scripts/assess_glyph_tail_adapter.py --stage assess`
generates the fixed 72 complete streams. The
[assessment protocol](../references/glyph_tail_assessment_protocol.md) fixes the
pilot target and uncertainty handling before results. Standalone inference uses:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. \
python scripts/infer_glyph_tail_language.py \
  --checkpoint artifacts/my_glyph_tail_run/fit/checkpoint_latest.pt \
  --image prompt.png --output artifacts/my_answer
```

That entry point accepts an image, weights and an output directory. It opens no
training or evaluation corpus and does not decode a lexical ID into a font.
The emitted image determines its own next feedback embedding. Historic-form,
open-ended language and efficiency claims still require their own evidence.
