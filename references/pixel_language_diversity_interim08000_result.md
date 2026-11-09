# Broader pixel-language training: the 8,000-update assessment

At 8,000 updates, the 512-question experiment reproduces 21 of its 64 assessed
training answers exactly, up from three at
[4,000 updates](pixel_language_diversity_interim04000_result.md). It still does
not demonstrate useful language generation for unseen prompts. None of the
24 normal development responses is pixel-exact, and qualitative inspection of
every response identifies no usable correct complete answer. Most outputs
contain broken or incomplete writing. These observations support improved
fitting of familiar answer images; they do not establish a general language
model.

This is a fixed interim assessment of the registered 16,000-update run. Its
architecture, data, objective and schedule remain unchanged. Final assessment
and independent feedback-encoder replay are pending. The assessment is not
used to select a checkpoint or extend the training budget.

| Fixed assessment group | Cases | Pixel-exact complete answers | Foreground F1 | Learned stops | Cap failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original training controls | 32 | 7 | 0.732672 | 32 | 0 |
| Added training examples | 32 | 14 | 0.819940 | 32 | 0 |
| Chinese development, normal | 16 | 0 | 0.353523 | 16 | 0 |
| Chinese development, question erased | 16 | 0 | 0.218205 | 16 | 0 |
| English development, normal | 8 | 0 | 0.030276 | 8 | 0 |
| English development, prompt erased | 8 | 0 | 0.044550 | 8 | 0 |

Each autonomous response starts from its original prompt PNG, observes its
own emitted images and uses learned stopping with the fixed 128-block cap.
Reference images and answer lengths do not enter generation. Five of the
32 assessed training pairs have both complete answers pixel-exact; none of
the eight Chinese development pairs does. Chinese normal-minus-erased F1 is
0.135318; English is −0.014274. The larger Chinese overlap difference partly
reflects lower erased-output overlap than at 4,000 updates. It is not a
semantic-correctness score or evidence that the factual answer is understood.

Whole-training correct-history image NLL is 0.020290, compared with 0.251025
at 4,000 updates and 9.749791 initially. All 512 training questions contribute.
The objective averages physical-piece NLL within blocks and questions; it is
not the unscaled joint likelihood of a full answer. The
[retained history review](evidence/pixel_language_diversity_interim08000/retained_exposure_review.json)
contains 32,000 example draws: 320 questions have 64 exposures and 192 have
60. Lower correct-history loss and better sampled training reproduction do
not establish correct autonomous answers on new inputs. The broader training
cohort also remains incompletely fitted.

All 24 normal development responses were viewed in source order on
[output-only sheets](evidence/pixel_language_diversity_interim08000/output_only/)
before the current reference comparison. The sheets preserve every original
output pixel and contain no reference answers. All 24 erased responses and
the first four examples from each training group were also viewed. The other
56 training responses were retained and checked numerically. These development
cases have been reviewed in earlier experiments, so the review is not fully
blinded or a sealed test. The
[source-first notes](evidence/pixel_language_diversity_interim08000/source_first_notes.json)
and [later qualitative comparison](evidence/pixel_language_diversity_interim08000/manual_review.json)
are separate records. There is no independent human grading or formal
semantic-accuracy estimate.

The two date-like responses previously readable at 4,000 updates now contain
damaged final digits. Partial readings are retained without reconstructing
them toward the reference:

| Development index | Approximate source-first reading | Required by the pictured passage | Interpretation |
| --- | --- | --- | --- |
| 12: year of the Treaty of Bern | 183?年 | 1874年 | Incomplete; the visible prefix disagrees |
| 15: Jurassic Coast World Heritage listing | 200?年 | 2001年 | No confident complete year |

The original outputs for
[index 12](evidence/pixel_language_diversity_interim08000/raw_outputs/development_0012_normal.png)
and [index 15](evidence/pixel_language_diversity_interim08000/raw_outputs/development_0015_normal.png)
are retained alongside every other response. The second output cannot be
called a correct complete year or confidently assigned a different complete
year. These two examples were chosen for discussion after inspection; they
are not a representative accuracy sample.

A separate [saved-prefix analysis](evidence/pixel_language_diversity_interim08000/prefix_diagnosis.json)
uses the existing emitted pixels and makes no neural-model calls:

| Training group | Cases | Exact first image | Exact complete answer | Exact first image followed by a complete-answer mismatch |
| --- | ---: | ---: | ---: | ---: |
| Original controls | 32 | 13 | 7 | 6 |
| Added examples | 32 | 23 | 14 | 9 |

Every normal and erased development response already differs from all allowed
reference first images. Thus drift from feeding back an earlier complete
output image cannot explain every development pixel error: an error is
already present before that feedback occurs. Autoregression within the first
image remains a possible contributor. Conversely, the 15 training responses
with an exact first image and an incorrect complete answer show that later
continuation also remains unreliable. Neither finding assigns semantic
accuracy or proves a single cause of failure.

The [independent CPU assessment review](evidence/pixel_language_diversity_interim08000/audit.json)
checks all 112 output tensors against their saved PNGs, reproduces complete
pixel scores and validates stopping/cap events and feedback-file hashes.
It does not re-encode feedback or replay the imported reader. All 112 raw
PNGs and 28 display sheets are preserved with
[retention hashes](evidence/pixel_language_diversity_interim08000/retention_manifest.json).
The [owning-state receipt](evidence/pixel_language_diversity_interim08000/owning_state_receipt.json)
binds the assessment to the 286 learned tensors captured at update 8,000.
The CPU watcher has preserved both fixed interim states and exited normally;
it makes no inference calls and selects neither state by quality.

The actor still includes a 7.54B-parameter pretrained body with 13.856M locally
trained parameters. Its inherited visual/language knowledge is not evidence
of a small foundation trained from glyphs from scratch. This trial establishes
neither matched-quality efficiency over an LLM nor historical-form or
long-form writing capability. English is untrained transfer in this Chinese
stage.

The next model-based diagnosis is already
[specified and implemented](pixel_language_first_block_diagnosis.md): after
the sole trainer exits, compare reference first-image likelihood under the
two questions about each passage, and compare each individual reference
likelihood with that of the saved greedy image. This separates evidence about
conditional preference from a possible greedy-decoding failure without
changing the active experiment. Its neural execution remains unvalidated and
no likelihood result is claimed here. The success criterion remains readable,
correct, complete answers to unseen image prompts.

The interim CPU review can be reproduced into a new directory:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/review_pixel_language_assessment.py \
  --run artifacts/pixel_language_diversity_training_v2 --update 8000 \
  --output artifacts/pixel_language_diversity_review08000_recheck
```

Training and final audit remain governed by the
[registered protocol](pixel_language_diversity_training_protocol.md).
