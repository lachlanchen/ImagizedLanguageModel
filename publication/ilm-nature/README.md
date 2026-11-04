# Language in continuous image space

This compact research article separates measured visual prediction, glyph
synthesis and conditional generation. Its reading layout follows the Lamp+LCD
V-SPICE v13.7 manuscript template: a two-column article, a matching review
copy, separate Methods and Supplementary Information, and evidence figures.
The earlier 96-page research record remains in `../ilm-image-native/`.

## Current result

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
| `ILM_Figures_review.pdf` | Verified fourteen-slide LibreOffice rendering |
| `frontmatter.tex`, `article_content.tex` | Article summary and scientific content |
| `figures/` | PDF, PNG and SVG evidence figures |
| `evidence/` | Reports with local paths normalized for delivery and source hashes |

The deck is checked through LibreOffice export and package inspection. Native
PowerPoint application playback is not part of that check. The title, narrative
and tables describe a research draft, not an accepted Nature publication.

## Rebuild

From the repository root, after the V53–V60 artifacts are present:

```bash
python publication/ilm-nature/generate_evidence.py
python publication/ilm-nature/generate_v54_evidence.py
python publication/ilm-nature/generate_v55_evidence.py
python publication/ilm-nature/generate_v56_evidence.py
python publication/ilm-nature/generate_v57_evidence.py
PYTHONPATH=. python publication/ilm-nature/generate_reading_interventions.py
PYTHONPATH=. python publication/ilm-nature/generate_reading_editable_figures.py
bash scripts/latex_build.sh publication/ilm-nature/manuscript.tex
bash scripts/latex_build.sh publication/ilm-nature/manuscript_review.tex
bash scripts/latex_build.sh publication/ilm-nature/supplementary_information.tex
```

Render the deck through LibreOffice, inspect it, and refresh the render receipt
before running `validate_and_sync.py`. That script refuses a deck or PDF whose
hash has changed since review, checks all nine embedded evidence workbooks and
verifies the delivered copies byte for byte.

`generate_evidence.py` checks the locked checkpoint identity before producing
values and figures. Source code, checkpoint hashes and frozen protocols are
recorded in the run report. Experimental code resides in `scripts/` and
`ilm/visual_lm/`. Data and model weights remain outside the publication package.

The fourteen-slide editable deck retains the earlier plates and adds native
block-reading, frozen-foundation and input-order charts, editable mechanism
shapes and actual answer images. Current
validated copies are delivered to Nutstore `Projects/ILM/` under the V60 version
folder and stable `LATEST` aliases. Local checksums verify copying; remote cloud
synchronization is not independently confirmed.

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
