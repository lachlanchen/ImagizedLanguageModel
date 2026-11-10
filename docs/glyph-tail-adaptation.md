# Contextual adaptation for image-language generation

The completed candidate trains the final four attention layers and the glyph
writer together. It gives **16/32 clear complete answers**, versus **11/32**
for the fixed-body control, on prospectively reserved Chinese questions.
Both miss the preset pilot and training-pixel gate. The
[paired result](../references/glyph_tail_paired_answers_result.md) retains
every answer and failure. This is partial language transfer, not a validated
usable general model. The main cache, fit, qualification and complete-answer
assessment have separate receipts. The earlier
[native-step comparison](../references/glyph_feedback_ceiling_result.md)
diagnoses feedback mismatch but does not explain all initial-glyph errors.

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

The original main fit stops at its cumulative memory-pressure wait allowance
after 2,926 updates. Recovery restores the complete update-2,000 optimizer
checkpoint and replays the 926 unsaved updates. Its actual four-update
checkpoint round trip is bit-identical across question and copying batches,
including all trained weights, optimizer tensors and CPU/CUDA random states.
Four extra verification updates are discarded. The
[recovery receipt](../references/evidence/glyph_tail_fit_recovery_v1/roundtrip.json)
and [execution protocol](../references/glyph_tail_fit_recovery.md) preserve the
failed attempts and resource amendment. Use `scripts/resume_glyph_tail_fit.py`
with CPU `--stage audit` before GPU `--stage run`, an unused recovery folder,
and no live fit or coordinator. The 18,000-step schedule and assessment remain
unchanged. An intermediate checkpoint is rejected by standalone inference.

Historical recovery progress remains in
`artifacts/glyph_tail_adaptation_v2/fit_recovery_02/status.json` and its
`history.jsonl`. The original failed attempts remain preserved; the completed
canonical `fit/status.json` now reports completion. Partial checkpoints are
training state, not qualified inference releases.

The fixed fit has now completed. An independent CPU
[checkpoint audit](../references/evidence/glyph_tail_final_fit_v2/checkpoint_audit.json)
confirms 18,000 real writer updates and 13,500 body/stopping updates: 13,500
question batches alternate with 4,500 writer-only copy batches. The frozen
encoder, decoder and normalization tensors are bit-identical to the starting
bridge. The resumed attempt peaks at 3.357 GiB allocated GPU memory; that
receipt covers 16,000 retained continuation updates and four discarded replay
checks, rather than the whole cache, fitting history or inherited pretraining.
Full-body qualification and complete answers still determine usefulness.

The matched broader-data control is prepared at
`artifacts/glyph_tail_readout_control_v2`. It uses the same cached questions
with G and stopping trained, and zero body-adapter updates. Its execution
resource allowance matches the resumed primary; the unused v1 preparation is
retained. Both final models must complete their raw-output review before either
reserved-reference score is opened. The
[control protocol](../references/glyph_tail_data_control_protocol.md) defines
the comparison and its interpretation.

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

The standalone command accepts ordinary raster image sizes and transparent
screenshots. It applies EXIF orientation, composites transparency onto white,
converts to grayscale, bounds large inputs and pads the right/bottom edges to
32-pixel alignment. `observed_prompt.png` records that preparation; the report
binds both original bytes and observed pixels. This preprocessing uses no OCR,
font or language interpretation. The registered assessment continues to use
its identical prepared images. Fixed numeric source-derived task controls
are authenticated before the body loads; they are not a glyph inventory.

`scripts/verify_glyph_tail_cold_inference.py` fixes its replay input as the
first reserved question, before outputs are known. After the complete assay,
it can run the real standalone CLI with empty working/model-cache directories,
audit file/network access and compare all output tensors, stopping, feedback
and PNG bytes. The [cold replay protocol](../references/glyph_tail_cold_inference_protocol.md)
separates that independence check from answer accuracy. A replay verification
has actually completed for `glyph_tail_adaptation_v2`, with all pixels,
states, stop decisions, feedback and PNG bytes identical. Its trace contains
no prohibited bank/font/corpus access or IP connection attempt. This result
is limited to the fixed first reserved input.

After both conditions are reviewed and scored,
`scripts/archive_glyph_tail_pair.py --output references/evidence/my_glyph_pair`
packages all 144 actual answer images, 52 review boards and losslessly packed
qualification pixels. It verifies all saved physical F1/exact scores and
copies the dataset notice. Failed cases remain present; large state tensors
and raw system-call traces remain local with their recorded hashes.

The subsequent [balanced writer protocol](../references/balanced_glyph_writer_protocol.md)
freezes the learned reader and gives first and later target positions equal
loss weight in larger batches. Its complete cache reproduces all 23,262 parent
training pixel scores and stopping counts. The 18,000-update refinement at
`artifacts/balanced_glyph_writer_v1` takes 156.38 s, including 20.02 s pressure
waiting, at 1.096 GiB peak allocation. The inherited model and earlier fitting
costs remain part of the method. An independent tensor audit confirms actual
optimizer counters and bit-identical frozen image modules and body adapters.
It passes physical qualification, but on a newly reserved context cohort gives
18/32 clear complete answers versus 19/32 for the unchanged parent; both give
2/32 canonical exact images. The one-answer difference is an uncertain initial
glyph, not an established semantic regression. All 144 new streams were viewed
before scoring. See the [complete result](../references/balanced_glyph_writer_result.md).
The original reserved cohort is development evidence and is not relabelled as
a fresh test for this repair. Further fitting the same states is not supported
as the sole remedy.
