# Full-page answer locations for a reading diagnostic

The image writer's completed normal-question comparison gives no OCR-exact
answer gain. A useful next diagnostic is whether its visual features make
the answer's position in the passage learnable. This separates a measurable
reading question from the demand to synthesize a new answer image. It does
not replace free image-language generation as the ILM objective.

There is already substantial source-copy evidence in this repository. The
[contextual span comparison](contextual_pixel_adaptation_result.md) obtains
26/64 held-out answers with adaptation versus 23/64 in control; both have
only 3/32 complete question pairs correct and preserve 32/32 new copies.
Those results do not qualify a natural reader. The present preparation
covers the current **complete 1024-square pages**, rather than the earlier
folded writing strips. No new learned localizer is implemented or evaluated
by the preparation.

## Actual data preparation

The [audit record](evidence/fullpage_answer_location_preparation.json) covers
all 288 existing natural QA pages: 256 training questions from 128 articles
and 32 held-out questions from sixteen disjoint articles. The original
renderer, font checksum, source text and image manifest are authenticated.
Re-rendering reproduces every existing input PNG pixel-for-pixel, and every
recovered line box matches the original recorded geometry.

Canonical answer occurrences are located only in the displayed passage.
All occurrences are retained: 36 examples contain multiple occurrences,
and 55 examples have at least one answer occurrence crossing a wrapped line.
Each occurrence has ordered line segments and physical horizontal advances.
The original page images are unchanged. The first training, first held-out
and first additional multiline examples were rendered with ground-truth
overlays and visually inspected. Those overlays are labelled as annotations,
never predictions.

The offline preparer sees source questions and references. Separate training
and validation annotation files contain numeric locations, source bindings
and grouping metadata; their image-only input index contains no answer,
location, passage length or target transcription. A future feature extractor
or deployed localizer must read that input index, not annotation metadata.

## Learning question and limits

For input image \(I\), let \(\mathcal S(I,Y)\) denote the annotated physical
occurrences of canonical answer \(Y\). A normalized distribution over source
locations can use the marginal loss

\[
\mathcal L_{\mathrm{loc}}(\theta)
=-\log\sum_{s\in\mathcal S(I,Y)}p_\theta(s\mid I).
\]

Reference access constructs supervision; the model condition remains the
input image. This expression is a proposed diagnostic objective, not an
executed training result. All candidate locations must be scored without
gold answer lengths, correct preceding answers or supplied answer locations
at inference. Held-out paired questions and erased-input controls remain
necessary to distinguish useful answering from layout shortcuts.

The rectangles are advance-coordinate envelopes. They do not establish
exact per-glyph segmentation or perfect source-image reconstruction: kerning,
overhangs and rasterization can affect a cut boundary. Location accuracy,
source copying and free image generation require separate measurements.
Even successful extractive answering would not demonstrate writing absent
from the prompt, historical semantics or broad independent language learning.

CMRC 2018 is a span-extraction reading dataset, which motivates this
diagnostic's scope. [Cui et al., 2019](https://aclanthology.org/D19-1600/).
Rendering questions as part of the input image also has prior precedent in
Pix2Struct; its image-to-text output differs from the ILM's image-output
objective. [Lee et al., 2023](https://arxiv.org/abs/2210.03347).

## Reproduction

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/prepare_fullpage_answer_locations.py \
  --output artifacts/fullpage_answer_locations_v1
PYTHONPATH=. python scripts/render_fullpage_answer_locations.py \
  --locations artifacts/fullpage_answer_locations_v1 \
  --output artifacts/fullpage_answer_locations_v1/review
```

Use fresh output directories. Preparation loads no neural model and performs
no optimizer update. Existing candidate generation and gradient tests remain
queued on the shared GPU0 schedule.
