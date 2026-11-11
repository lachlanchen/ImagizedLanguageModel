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
