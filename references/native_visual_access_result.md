# Visual copying improves; natural question answering fails

The completed 4,096-update experiment gives **27/32 exact stopped held-out
copies and 0/32 correct natural answers** through the frozen auxiliary native
text head. English copies are 16/16; Chinese copies are 11/16. This is useful
short-string reading progress, without reliable Chinese copying, page-level
question understanding or independent image-answer generation.

All 32 natural answers collapse to two years: 27 are `1999年` and five are
`1998年`. They are incorrect for the actual questions, rather than alternative
correct wording. In the separately specified paired controls, **15/16 passage
pairs receive the same answer to their two different questions**. Removing
the question leaves **29/32** normal outputs unchanged. No passage pair has
both answers correct; none of the sixteen shared erased-page outputs matches
either reference. The visual-adapter recipe fails its language requirement.

| Auxiliary text readout | Before | After |
| --- | ---: | ---: |
| Fixed screen: four training copies | 0/4 | 4/4 |
| Fixed screen: four held-out copies | 0/4 | 3/4 |
| Fixed screen: four held-out natural questions | 0/4 | 0/4 |
| Separate full validation: copies | Not measured | 27/32 |
| Separate full validation: natural questions | Not measured | 0/32 |

The full observation includes the complete declared held-out cohort. Repeated
screen cases are retained separately and never pooled as independent examples.
The public CMRC data have already been used in exploratory project work; this
is not a sealed benchmark or replication of the source authors' evaluation.

## What was learned and measured

The complete native Show-o2 HQ source has 2,830,777,056 parameters plus a
126,892,531-parameter Wan codec. Gated rank-16/alpha-16 residuals add
8,489,984 trainable parameters across 158 semantic visual-encoder and fusion
linear layers. The source weights, language body/head and image generator stay
frozen. The adapted native model therefore contains 2,839,267,040 parameters,
plus its codec. External pretraining remains part of the model's provenance.

All 128 prepared training copies and 256 training questions enter the fresh
run, alternating shuffled cyclic lists for 4,096 updates. All 384 IDs occur;
no validation ID is optimized. Every optimizer record has finite loss and
gradient norm. The single invocation takes **2,758.904 seconds** with
**6.894 GiB peak allocated memory** on GPU 0, an RTX 4090 D. These values
exclude source pretraining/acquisition, input caching and separate evaluations.
The 64-call full assessment takes 26.916 seconds; the 48-call paired
assessment takes 33.030 seconds, each excluding source loading. Low memory
and completed execution do not establish useful language efficiency.

Changing prompt content enters through complete source pixels. Correct
preceding answer labels are used only in the auxiliary training loss.
Autonomous evaluation loads fresh input images and numeric adapter weights,
without training cache, reference answers or correct answer histories. It
still uses the native language vocabulary and output head. **These are text
readouts, not images written by the trained learner.** The direct continuous
image-answer stage remains unimplemented in this route.

The first four held-out copy readouts include one Chinese substitution:
`语树石字光河画山雪羊` becomes `语树石字光河画山雪字`. Across the full set,
five Chinese copies substitute, insert or omit characters. Exact scoring uses
NFC, removes Chinese whitespace and collapses English whitespace, while
preserving case, punctuation, quotes and every written character. Native EOS
is required for exact-stopped scores. Character-multiset F1 is retained only
as a descriptive, order-insensitive statistic.

The question intervention preserves every context pixel and erases the
question paragraph through the bottom, including the constant answer marker.
Both erased versions of each passage are pixel-identical. Latent/readout seeds
are matched per passage across its two questions and one shared erased page.
There are 48 independent calls, not 64. Output changes alone cannot establish
correct answers: only one normal pair differs, and all normal answers fail.

## Evidence and decision

The [complete export](evidence/native_visual_access_result/result.json)
binds 179 files: all raw screen/full/intervention readouts and reports, complete
input/reference pages, sixteen erased pages and the 4,096-step optimizer
journal. The final checkpoint SHA256 is
`e2937d2fa986fac0e7ca1964607d5dca06194b698b6a25ad0644635f585427fb`.
Weights and runtime caches remain outside Git. Source diagnostics, including
assisted image-writing failures, remain [separate](native_unified_source_result.md).
CMRC passage images and references retain CC-BY-SA-4.0 attribution to
Cui et al., EMNLP-IJCNLP 2019; copy strings are synthetic. Native source/model
are NUS Show Lab, Apache-2.0.

The visual input path can learn new short strings, especially English within
this prepared distribution. It has not learned to select answers from complete
Chinese passages. Scaling this answer-only visual-adapter recipe or starting
image-writing adaptation would not resolve that measured prerequisite by
itself. A subsequent experiment must test better visual-language alignment
and question-dependent selection, retain these failed controls, and separately
demonstrate direct image answers. The general ILM goal remains unachieved.

[Learning protocol](native_visual_access_learning_protocol.md),
[assessment protocol](native_visual_access_assessment_protocol.md) and
[execution guide](../docs/native-visual-access.md) preserve the exact workflow.
