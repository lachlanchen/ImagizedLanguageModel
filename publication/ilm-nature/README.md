# Language in continuous image space

This compact research article separates measured visual prediction, glyph
synthesis and conditional generation. Its reading layout follows the Lamp+LCD
V-SPICE v13.7 manuscript template: a two-column article, a matching review
copy, separate Methods and Supplementary Information, and evidence figures.
The earlier 96-page research record remains in `../ilm-image-native/`.

The [change-marked article](redlined_manuscript.pdf) and
[supplement](redlined_supplementary_information.pdf) compare the current
revision with the preserved V68 conditional-writing baseline. Their
[builder](build_semantic_transfer_redlines.py) uses pinned latexdiff and inlines the
abstract so its changes are visible. The earlier supplement-only redline and
builder remain inside the named baseline.

## Current result

Numeric source-derived task controls retain bounded image reading in the
imported vocabulary-free contextual body: four of eight checked Chinese
answers are exact at the declared native sampling budget, versus zero with
question ink erased. A separate temporary text projection produces these
diagnostic answers. All 742 helper projection decisions replay exactly;
the whole visual forward pass is not independently replayed. This is reader
evidence, not successful image writing. Supplementary Note 29 reports the
full comparison and limitations.

A 38.65M-parameter causal joint-flow writer is implemented with prompt-only
visual memory and re-encoded physical-output feedback. Real-example execution
checks pass, including exact optimizer/RNG resume. Full-corpus caching,
40,000-update training and the original autonomous audit are in progress.
Supplementary Note 30 gives the complete mathematical and deployed path;
it reports no new language accuracy. Figure S10 shows the measured reader
comparison and its helper boundary; Figure S11 shows the implemented training
and physical-feedback computation. Both are native editable slides. The deck
now has 26 slides, preserving all 24 earlier slides and their canonical PDF
appearance. The [current figure verifier](verify_semantic_transfer_figures.py)
checks source scores, chart workbooks, native parts and rendered figures.
The [current publication builder](build_semantic_figure_publication.py) rebuilds
the article, supplement and matching redlines.

A complete conditional-writing actor finishes 10,000 updates on all 12,976
admitted Chinese/English instruction examples. It adapts 9.352M of 7.539B
parameters on one RTX 4090 D, peaking at 16.965 GiB allocated memory. The fixed
autonomous audit fails: 0/128 exact Chinese answers, 32.671% foreground F1 and
0/64 complete question pairs. Question pixels have a resolved 13.163-point
effect [10.933, 15.730], and all 128 normal Chinese outputs stop through the
learned decision. Three answer-quality requirements fail; input dependence
and stopping do not qualify language. Figure 10 and Supplementary Note 27
show the original six decisions and the complete first two raw outputs.

A frozen 32-case diagnosis finds poor prediction even with exact history
and known answer length, including on training examples. Supplied-code F1 is
99.95–100%, while assisted prediction reaches only 21.8–35.6%. No first predicted
block is exact. Figure S9 and Note 28 localize a failure preceding generated
history without identifying a unique repair. The diagnostic makes no optimizer
updates; CPU verification checks retained tensors and pixels rather than
independently replaying the native neural forward pass. Development data are
reused; sealed evaluation remains unopened. External foundation pretraining
and prior codec learning are outside the reported local adaptation cost.

A separate continuous output space qualifies supplied-code reconstruction.
The original frozen writing codec reaches 99.998% foreground F1 and 126/128
exact development blocks, versus 21.413% F1 with shuffled codes. The broader
direct writer fails fidelity at 56.563% and 3/128 exact blocks, despite fitting
63/64 training blocks in a smaller diagnostic. Figure 9 and Notes 24–25 show
the actual training/development distinction and the first fixed raw images.
The two native charts retain editable, source-matched workbooks.

The complete actor connects contextual states to those separate writing
coordinates. It retains 7.539B total parameters, trains 9.352M, and passes all
24 GPU execution checks at 16.953 GiB peak allocated memory. Figure S8 and
Note 26 give the image-only feedback loop, detached target-code supervision,
loss and numerical checks. Two-update output remains fragmented. These are
execution and reconstruction results; autonomous language is unqualified.
That revision extends the deck to 24 slides, preserving all preceding 22 slide
renders and original slide/chart parts. Three native charts have source-matched
embedded workbooks. Earlier source figures remain unchanged.

A matched question--answer assignment term weakens free image writing.
Control and aligned actors each train 490M parameters for 2,000 updates from
identical weights and sample draws. Exact answers are 2/256 and 1/256; neither
arm has a fully correct pair among 393. Raster F1 is 3.229% and 2.417%, with
aligned minus control -0.812 percentage points, 95% article-bootstrap interval
[-1.504, -0.202]. Both question effects resolve positively, but both absolute
reading gates fail. Figure 8, Supplementary Note 22 and native slides 18–19
show the method, paired uncertainty and first two unrepaired answers.

A separate inherited 7.548B actor passes complete execution checks at
17.181 GiB with 25.58M trainable parameters and no deployed vocabulary module.
Its writer-only 10,000-update test fails reconstruction: 34.000% foreground F1
under correct image conditions versus 21.420% shuffled, zero exact blocks.
This does not qualify prompt-to-answer language. Figure S7, Note 23 and native
slide 20 distinguish the executed mechanism from language capability. All
256 raw writer image scores were independently recomputed. Local 4090 work
does not include external foundation pretraining.

A matched conditional bitmap mixture improves pixel fit while failing the
language gates. K=8 lowers held-out NLL to 0.1248 versus 0.1542 for K=1 and
raises strict-raster F1 from 0.234% to 0.815%. Both have zero exact primary
answers and zero both-correct question pairs. The K=8 question effect remains
unresolved, so the registered density intervention and both reading gates
fail. Both standalone offline bundles reproduce their outputs exactly.
Figure 7, Supplementary Note 21 and native editable slides 16–17 give the
probability law, mechanism, full comparison and first two raw answers.
The complete V61 package is preserved in `baselines/v61/`.

The auxiliary vision reader passes reference calibration but returns three
invalid actor responses, so its original fixed-case diagnostic fails.
An explicitly post-hoc table in Supplementary Note 21 bounds every missing
score without dropping cases: K=8 has zero exact transcript agreements and no
resolved character-accuracy question effect. The primary rejection is unchanged.
The earlier primary-only V62 package is preserved in `baselines/v62-primary/`.

Full native image-language adaptation now provides a free-writing test. All
477.15M externally pretrained parameters receive answer-image gradients for
1,000 updates on one RTX 4090 D, taking 1,501.1 seconds at 9.912 GiB allocated
memory. Held-out pixel NLL improves from 0.6842 to 0.1548 nats, yet exact answers
remain 0/256 and both-correct pairs 0/393. Raster F1 is 0.149%; raw answers
remain fragmented. Five of ten gates pass, and the registered reading decision
is rejection. Independent native-PNG export/replay passes. Figure 6,
Supplementary Note 20 and editable slide 15 report the failure without output
repair. V60 is preserved in `baselines/v60/`.

Complete observed image blocks improve answer structure. The 19.60M model
answers 25/256 questions exactly (9.766%), versus 3/256 blanked-question and
21/256 matched additional glyph-answer training. Raster F1 is 27.033%; its
4.893-point gain over the matched control has interval [0.469, 9.047], while
the exact-answer difference remains unresolved. Only 4/393 paragraph pairs
have both answers correct, failing the registered paired gate. Figure 5,
Supplementary Note 17 and slide 12 show the law, scores and raw first answers.

Two registered interventions also fail. Frozen MIXAR features give 6/256
exact answers and 11.588% F1, versus 24/256 and 25.340% on frozen student
features. This shifted-interface probe does not rank native encoder quality
or count external pretraining as local work. Moving the question first gives
21/256 exact answers and 23.765% F1, versus 27/256 and 26.908% for matched
context-first continuation; the difference is unresolved. Supplementary
Notes 18–19, Figures S5–S6 and slides 13–14 report both negative results.
All local phases use only physical GPU 0. The general language goal remains
unachieved. The V57 baseline is preserved in `baselines/v57/`.

Grounded visual reading now provides a direct language-use test. The 19.01M
copy/raster model answers 10/256 held-out natural questions exactly, versus
1/256 with question pixels blanked and 0/256 for matched no-copy training.
Raster F1 is 15.718%; its question-dependent gain is 10.165 points
[7.043, 13.371]. Only 1/393 shared-paragraph question pairs has both answers
correct. The reading model fails three of seven gates and remains rejected.
Figure 4, Supplementary Note 16 and slide 11 show the full comparison and
unselected raw failures. V56's readable baseline remains in `baselines/v56/`.

With unchanged V55 weights, raw-pixel beam search improves exact next-image
match from 8.252% greedy to 11.621% on 2,048 prefixes. Earlier-history shuffling
reduces beam accuracy to 9.863%. All beam outputs exactly match a canonical
form, but eight-step visible feedback repeats adjacent images at 25.446%,
against 0.112% in the observed futures. The decoder fails its rollout gate and
remains rejected. Supplementary Notes 14–15, Figures S3–S4 and slides 9–10
separate form validity, immediate choice and sequence failure. V55's readable
baseline is preserved in `baselines/v55/`.

Joint reader–density learning improves held-out exact image NLL to 7.026
nats/glyph versus 7.579 for a matched frozen-reader continuation. The paired
gain is 0.553 nat, 95% source-record interval [0.450, 0.665]. Both arms start
from identical V54 EMA weights and make 8,000 additional updates. Joint training
takes 342.4 seconds at 0.648 GiB allocated CUDA memory on one RTX 4090 D,
excluding earlier foundation training. Greedy supplied-image copying remains
exact on 256/256 targets.

Autonomous language remains unqualified: joint first-sample identity is 9/256
ordered versus 8/256 earlier-shuffled and 6/256 for the matched frozen control.
The joint model passes six of eight unchanged gates, failing sampled quality
and ordered generation. Supplementary Note 13, Figure S2 and the final two
editable slides report both arms and raw images. The readable V54 baseline is
preserved in `baselines/v54/`.

## Exact-density baseline

An exact autoregressive raster density now preserves all 256 supplied images
under greedy copying, with stochastic-copy F1 0.99949. The independent runtime
has 18.71M parameters, with 2.43M new density parameters trained for 8,000
updates in 207.8 seconds on one RTX 4090 D (0.400 GiB peak allocated memory,
excluding earlier reader training).

Autonomous language still fails the locked audit. Natural-target NLL is
15.219 nats/image ordered versus 15.254 shuffled; the paired advantage has
95% source-record bootstrap interval [-0.1014, 0.1693]. First-sample identity
is 4/256 ordered versus 3/256 shuffled. The post-generation evaluator contains
9,932 distinct images and covers every target. Copying and low teacher-forced
image NLL do not establish useful language generation.

The earlier direct diffusion result remains reported: 22.18M total parameters,
8,000 updates in 361.6 seconds, first-position identity 3/256 ordered and
shuffled, and block identity 0.586% versus 0.488%. Figure 3 adds the exact-density
result without replacing Figures 1–2. Compact V53 article/review/supplement
baselines remain in `baselines/v53/`.

## Reading and editing

| File | Purpose |
| --- | --- |
| `manuscript.pdf` | Compact reading copy |
| `manuscript_review.pdf` | Matching line-numbered review copy |
| `supplementary_information.pdf` | Complete method, metric limits and reproducibility |
| `ILM_Figures_editable.pptx` | Native diagram shapes and bar chart with embedded workbook; raw glyphs remain images |
| `ILM_Figures_review.pdf` | Verified twenty-four-slide LibreOffice rendering |
| `redlined_manuscript.pdf`, `redlined_supplementary_information.pdf` | Matching change-marked review copies |
| `frontmatter.tex`, `article_content.tex` | Article summary and scientific content |
| `figures/` | PDF, PNG and SVG evidence figures |
| `evidence/` | Reports with local paths normalized for delivery and source hashes |

The deck is checked through LibreOffice export and package inspection. Native
PowerPoint application playback is not part of that check. The title, narrative
and tables describe a research draft, not an accepted Nature publication.

## Rebuild

From the repository root, after the corresponding source artifacts are present.
Earlier figure builders below reproduce their historical packages; the final
V68 builder appends to the immutable V67 baseline:

```bash
python publication/ilm-nature/generate_evidence.py
python publication/ilm-nature/generate_v54_evidence.py
python publication/ilm-nature/generate_v55_evidence.py
python publication/ilm-nature/generate_v56_evidence.py
python publication/ilm-nature/generate_v57_evidence.py
PYTHONPATH=. python publication/ilm-nature/generate_reading_interventions.py
PYTHONPATH=. python publication/ilm-nature/generate_native_pixel_evidence.py
PYTHONPATH=. python publication/ilm-nature/generate_native_pixel_editable_figures.py
PYTHONPATH=. python publication/ilm-nature/generate_correlated_density_evidence.py
PYTHONPATH=. python publication/ilm-nature/generate_correlated_density_editable_figures.py
PYTHONPATH=. python publication/ilm-nature/generate_transcript_bounds_evidence.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python publication/ilm-nature/generate_assignment_evidence.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python publication/ilm-nature/generate_assignment_editable_figures.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python publication/ilm-nature/generate_codec_language_figures.py
bash scripts/latex_build.sh publication/ilm-nature/manuscript.tex
bash scripts/latex_build.sh publication/ilm-nature/manuscript_review.tex
bash scripts/latex_build.sh publication/ilm-nature/supplementary_information.tex
CUDA_VISIBLE_DEVICES='' python publication/ilm-nature/build_codec_language_redlines.py
```

Render the deck through LibreOffice, inspect it, and refresh the render receipt
before running `validate_and_sync.py`. That script refuses a deck or PDF whose
hash has changed since review, checks the twenty native charts and their preserved
or source-matched workbooks, and
verifies the delivered copies byte for byte.
`PYTHONPATH=. python publication/ilm-nature/verify_native_pixel_rasters.py`
also checks the six PDF and four PowerPoint raw images against their source
bits and confirms that the prior fourteen slides remain unchanged.
`PYTHONPATH=. python publication/ilm-nature/verify_correlated_density_rasters.py`
checks six additional PDF and six PowerPoint rasters against the V62 source
bits and confirms that all fifteen V61 slide XMLs remain unchanged.

`generate_evidence.py` checks the locked checkpoint identity before producing
values and figures. Source code, checkpoint hashes and frozen protocols are
recorded in the run report. Experimental code resides in `scripts/` and
`ilm/visual_lm/`. Data and model weights remain outside the publication package.

`CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python publication/ilm-nature/verify_assignment_publication.py`
checks six source-exact PNG, PDF, PPTX and deck-PDF images, both new chart caches,
workbooks and paired intervals, and preservation of every earlier deck part
except global slide registrations. It also compares earlier slide rendering.
Use lossless PDF image compression and disable resolution reduction when
rendering the deck. The Artifact Tool runtime is unavailable locally; the
new builder edits OOXML directly without python-pptx authoring.

The twenty-four-slide editable deck retains earlier plates and adds native
block-reading, frozen-foundation, input-order and conditional-density charts,
editable mechanism shapes and actual answer images. Current
validated copies are delivered to Nutstore `Projects/ILM/` under the
`ILM_NatureStyle_20261105_V68_Conditional` version
folder and stable `LATEST` aliases. Local checksums verify copying; remote cloud
synchronization is not independently confirmed.

The current verification entry point is
`CUDA_VISIBLE_DEVICES='' python publication/ilm-nature/verify_codec_language_publication.py`.
It binds all three current evidence receipts to pinned hashes, verifies the
229-file baseline against its source Git commit, checks all 22 prior rendered
slides, compares the three new editable charts with their source numbers and
matches every displayed writing raster in the deck, standalone figures,
article, review and supplement. Historical validators retain their original
slide-count assumptions and should be run on their corresponding baselines.
Lossless LibreOffice export disables image-resolution reduction. Artifact Tool
is unavailable locally; native OOXML authoring and LibreOffice rendering are
used, with no native PowerPoint playback claim.

The independent checkpoint export and native-image CLI are described in
[`exact_raster_density_v54_result.md`](../../references/exact_raster_density_v54_result.md).
The output retains the scope of the tested task. Chinese instruction following,
historical writing, etymology, multiple fonts, page-scale generation and
capability-matched text-model efficiency remain open requirements.

[Training and native-image inference guide](../../docs/exact-raster-language.md)
and [matched V55 result](../../references/joint_exact_raster_v55_result.md)
give complete code entry points, checkpoint hashes and qualification limits.

Supplementary Note 12 and Figure S1 report a separately planned, post-result
context diagnostic. Swapping the entire condition between source records
raises NLL to 20.630 nats/image; the ordered advantage is 5.411 nats with
95% interval [4.514, 6.416]. Earlier-history shuffling still has no resolved
whole-image effect. Context sensitivity does not reverse the failed generation
decision. Raw row losses and the reproducible diagnostic entry point are
described in [`v54_context_attribution_result.md`](../../references/v54_context_attribution_result.md).
