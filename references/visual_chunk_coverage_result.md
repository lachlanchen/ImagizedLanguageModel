# Visual spans and language-state updates

The frozen source reader gave 12 of 16 exact answers with native updates and
3 of 16 with character updates. That comparison changed the input vectors
and sequence of updates together. The next matched comparison should keep
the source update positions fixed and compare exact vectors with learned
encodings of the corresponding complete glyph images. The result will test
whether the image encoder preserves contextual decisions under the same
update sequence. See the [feedback comparison](glyph_feedback_ceiling_result.md).

The present audit uses the authenticated local teacher's tokenizer and
existing canonical training rasters. It measures whether each source span
can be represented by these pixels. A span must decode completely, render
each character and have a pixel sequence unique among source units.

| Training material | Examples | Complete and unambiguous | Also at most eight glyphs per span |
| --- | ---: | ---: | ---: |
| Actual inherited CMRC answers | 3,176 | 3,103 | 3,101 |
| Distinct original training paragraphs | 1,128 | 580 | 523 |

The 3,103 eligible answers use 17,286 native updates across 22,597 glyphs.
That is a 23.5% reduction in contextual update count relative to a single
update per glyph on these answers. Actual GPU cost also includes the image
encoder and writer. The native text baseline already makes those 17,286
updates. The result therefore sets a useful architecture comparison and
leaves any overall efficiency advantage to measurement.

The lower paragraph coverage exposes a limitation of one span per source
unit with the existing font inventory. Some native units contain only part
of a UTF-8 character, others lack a canonical glyph, and 41 units across
the source vocabulary share their complete pixel sequence with another
unit. A whole sequence can still decode correctly when separate units
contain incomplete characters. Group those units at complete character
boundaries during data preparation, and preserve their original teacher
update range. That lets the model learn the relationship between one visual
span and several continuous state updates.

## Why averaging glyph embeddings is insufficient

For 2,294 unambiguous multi-glyph units seen in the inherited answers, the
mean of the exact single-glyph anchor vectors has median cosine similarity
0.146 to the native unit vector. The first glyph gives 0.095. Median relative
L2 error for the mean is 1.162. Learn the span encoder from the complete
visual pattern and test it through the contextual body. See the
[measured vector audit](evidence/visual_chunk_coverage_20261111.json).

The structural constraint is simple. If two source units render to the same
image span \(X\) and have different teacher vectors \(w_a,w_b\), an encoder
with input \(X\) must produce the same output for both. Under squared error
and probabilities \(p,1-p\), its optimum is the conditional mean. Its minimum
expected error is

\[
  p(1-p)\lVert w_a-w_b\rVert_2^2.
\]

Context can resolve some ambiguous readings. For the remaining ambiguity,
the model can learn the distribution of suitable visual continuations from
actual contexts. The inverse of an information-losing raster cannot uniquely
recover a hidden spelling. An injective rendering retains the source string's
discrete entropy. Potential savings come from where and how often the model
updates its state and how much computation it uses at each update. See the
[readout and state analysis](glyph_readout_and_clock_math.md).

## Next model experiment

Prepare whole passages and complete answer sequences from authentic training
material. Supply the exact context targets from an offline teacher prefill.
Use actual image-prefill states to check transfer to prompts read through the
visual front end. Learn a causal body and a shared span encoder and writer.
Keep source unit IDs and rendering assets on the preparation side. The
deployed loop consumes the pixels it emits and maintains continuous state.

Compare exact source feedback with learned span feedback under the same
positions. Then compare full generated answers and continuations, including
error accumulation and stopping, using the pixels actually emitted. The
7.522B source body is the reference for this test. A smaller standalone
visual body and measured quality at comparable compute remain the goal.
The audit used training sources only and performed no neural fit or
generation. See the [engineering goal](../docs/first-imagized-language-model-goal.md).
