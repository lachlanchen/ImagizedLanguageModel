# Historical reading and writing from attested images

Use the sealed families of the content-grouped historical partition. Select
the forms by a fixed hash order. Exclude exact rasters eligible for historical
training under the original V34 partition. Exclude reading examples with
ambiguous modern labels among identical sealed rasters. Verify the selected
source SVG and its registered binary raster. The catalog's entry, stage and
glyph label accompany every evaluation reference. See the
[historical partition](historical_glyph_content_split_result.md).

The suite contains two tasks and 36 normal prompts.

| Task | Cases | Evidence and requested output |
| --- | ---: | --- |
| Historical reading | 18 | One historical image. Write the recorded modern traditional character. |
| Stage selection | 18 | Two labelled historical images from one catalog entry. Write the image of the requested stage. |

The reading cases cover six oracle, six bronze and six seal forms. Each
reading case has a control with its historical tile erased. The stage
selection cases form nine pairs, three for each combination of stages. Both
questions in a pair share the same source images and labels. Padding equalizes
question length, so their question-erased controls have identical pixels.
Retain all 72 prompt PNGs and separate the reference images from inference.
Historical input and output shapes are identified by catalog image hashes.

Generate each answer with the same image-feedback model and stopping rule
used for modern writing. Use a four-glyph safety cap for these requested
single-glyph answers. For stage selection, record exact pixel reproduction
and foreground F1 against the requested form. Compare the generated image
with the other form from the same input. Count pairs where both stopped
single-glyph answers are uniquely closer to their requested forms. A separate
blind reading is needed to judge modern answers that differ from the canonical
font. The stage-selection task measures use of supplied historical evidence.
Reconstruction from memory and a full etymology explanation remain separate
tests.

Measure the fixed codec directly on the historical source images before
interpreting autonomous failures. Encode each source tile and decode its
continuous latent. Also measure the five tall bronze crops as four 32×32
tiles each. These crops preserve strokes that disappear in square 32×32
rasters. Assemble the decoded tiles in their original vertical order and
score the whole crop. Report the reconstruction measures separately from
answer correctness. See the [resolution evidence](historical_glyph_raster_addendum_result.md).

Run the suite after obtaining a trained compact checkpoint. Pass its input
manifest to `scripts/infer_compact_visual_span.py` with `--maximum-glyphs 4`.
Use `scripts/eval_historical_visual_transfer.py answers` for the resulting
answer images. The `codec` stage measures the fixed physical representation.

## Measured source and codec fidelity

The 36 normal prompts use 27 distinct historical rasters from 18 sealed
source families. All 72 images reload to the original tiles. Each of the
nine stage-selection pairs has byte-identical question-erased controls.
The selected historical rasters have zero exact overlap with the original
V34 historical training partition. Near-duplicate exposure is unknown.

| Representation | Forms | Exact codec round trips | Mean foreground F1 |
| --- | ---: | ---: | ---: |
| Oracle, one 32 × 32 tile | 6 | 6 | 1.0000 |
| Bronze, one 32 × 32 tile | 9 | 9 | 1.0000 |
| Seal, one 32 × 32 tile | 12 | 5 | 0.9958 |
| Tall bronze, four tiles | 5 | 0 | 0.9389 |

The single-tile panel shows the original raster and its decoded form for
each selected image. The tall panel shows breaks in thin strokes after
decoding and vertically reassembling the four tiles. Both use the same
authenticated codec as the compact language model. See the
[single-tile panel](evidence/historical_visual_historical_32_20261111.png)
and [tall bronze panel](evidence/historical_visual_bronze_tall_128_20261111.png).

The sparsest selected source, bronze glyph `B12643`, retains ten ink pixels
at 32 × 32. The codec reproduces those pixels exactly. Rendering the same
authenticated SVG at 128 × 128 shows the longer strokes and contains 407 ink
pixels. This post-hoc inspection identifies lost detail in the input raster.
The case remains in the registered suite. Historical reading and autonomous
answer generation await the trained checkpoint. See the
[resolution panel](evidence/historical_visual_sparse_source_resolution_20261111.png)
and [measured evidence](evidence/historical_visual_evaluation_20261111.json).
