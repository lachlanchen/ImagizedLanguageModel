# Native vision reads some partial answers, but does not complete the ILM

8 November 2026. A frozen native visual reader can use answer-prefix pixels:
at the fixed middle positions, it gets the next visible codepoint right on
**12/12 copying requests and 23/32 Chinese QA questions**, versus **0/12 and
2/32** after prefix erasure. It nevertheless stops on **0/44 complete visual
prefixes**. A separately cross-fitted numeric state map improves agreement
with the text teacher but does not provide reliable Chinese content or
stopping. Both declared writer-admission rules fail. No new image-answer
model is qualified.

The [reading protocol](visual_history_reading_protocol.md),
[alignment protocol](visual_history_bridge_protocol.md),
[complete receipt](evidence/visual_history_reading_result/result.json) and
[all 388 raw responses](evidence/visual_history_reading_result/README.md)
retain every failure and control. The [execution guide](../docs/visual-history.md)
reproduces both stages.

## Native visual-history probe

Every input contains the original question pixels above a separator and a
lower panel labelled as the answer already written. The lower panel receives
a correct reference prefix, including the empty and complete cases. The
prefix is reread through the actual visual encoder; it is not inserted as a
learned substitute for a native language-history embedding. No remaining
answer is supplied as history. Answers are short extractive/copy targets, so
their content can already appear in the question or passage.

The cohort is the previously used twelve copying requests and 32 seen Chinese
QA questions. Distinct empty, middle, one-codepoint-short and complete prefix
positions yield 172 paired visual/text calls, plus 44 middle-prefix erasures:
388 calls in total. Correct history is supplied, so this is an assisted
diagnostic on reused public cases, not an autonomous image rollout or sealed
language benchmark. Its text projection, token-row access and tokenizer are
offline instruments, not a deployed ILM language head.

| Route | Exact missing suffix | First codepoint, nonempty suffix | Complete prefixes empty and stopped |
|---|---:|---:|---:|
| Copy, visual page | 27/48 | 30/36 | 0/12 |
| Copy, native text control | 41/48 | 29/36 | 12/12 |
| Copy, middle prefix erased | 0/12 | 0/12 | Not applicable |
| Chinese QA, visual page | 43/124 | 52/92 | 0/32 |
| Chinese QA, native text control | 77/124 | 82/92 | 8/32 |
| Chinese QA, middle prefix erased | 1/32 | 2/32 | Not applicable |

Scoring trims only outer whitespace. Quotes, explanations, punctuation and
internal whitespace remain. For example, an empty visual prefix produces
`“春风吹过河岸”`, which fails the unquoted reference. With `春风吹` visible,
the reader returns exactly `过河岸`; erasure returns the whole quoted phrase.
With the complete phrase visible, it repeats a phrase instead of stopping.
All three outcomes remain. Middle-prefix first-codepoint gains are 100 and
65.625 percentage points on copying and QA, but the four content/stopping
floors fail; only the two prefix-effect floors pass.

The source text control also fails most QA terminal events. Matching that
teacher cannot guarantee correct stopping. These results support a bounded
capability to read and continue supplied visual history; they do not establish
an appropriate state at every step, semantic completeness or an image writer.

## Cross-fitted representation learner

An affine residual is fitted between paired initial visual/text states after
unit-RMS normalization. The fixed ridge penalty is .1. Leave-one-pair-out
cross-fitting excludes both requests/questions and every prefix of the held
pair. There are 22 folds: six copying pairs and sixteen QA paragraph pairs.
Erased states never enter fitting. Every one of 216 visual/erased calls gets
one held-out prediction. Only numeric states and group membership enter the
fit; remaining-answer strings are used afterward for evaluation.

| Readout | Copy teacher first-piece agreement | QA teacher first-piece agreement | Copy next-codepoint agreement | QA next-codepoint agreement | Correct complete/one-short stopping pairs, copy / QA |
|---|---:|---:|---:|---:|---:|
| Identity | 27/48 | 61/124 | 30/36 | 52/92 | 0/12; 0/32 |
| Mean residual | 29/48 | 63/124 | 30/36 | 54/92 | 2/12; 0/32 |
| Learned ridge residual | 37/48 | 65/124 | 30/36 | 54/92 | 4/12; 0/32 |

This stage scores a **single native piece**, not a new complete continuation.
Incomplete Unicode pieces receive no codepoint credit. The denominators
differ from exact suffix scoring above. Copy teacher agreement rises by
20.833 percentage points, but copy content does not improve. QA gains two
canonical first-codepoint matches, also achieved by the mean control. Reduced
state error therefore does not demonstrate better conditional language.
Six of eight alignment-admission floors fail. The small numeric maps contain
about 1.35 million parameters, excluding the large externally pretrained
reader and temporary vocabulary projection. They are cross-fitted research
components, not complete lightweight ILMs.

## Integrity, costs and next constraint

The original probe takes **662.957 seconds**, including **126.201 seconds**
setup, and peaks at **20.051 GiB** allocated memory on GPU0. The 388 complete
calls take 536.756 seconds after setup. External pretraining is excluded.
CPU verification reconstructs every physical page and erasure, recompiles
numeric controls and replays all **2,642 native projection decisions** from
retained states. It does not rerun the whole visual/contextual body.

The final 22 analytic fits and factor checks take 2.823 seconds; the complete
CPU fit/readout stage takes 47.976 seconds, including archive access and
temporary projection setup. There are no gradient-optimizer updates or new
source-body calls in this learner. Three earlier fits stopped before readout
scoring at an overly strict float32 multiplication-order check. Their saved
parameter tensors equal the corresponding final-fold tensors exactly. The
correction changes numerical verification, not the learner or quality gates.
The largest float64 affine-equivalence error is 9.24e-14; the retained float32
forward differs from its double-precision calculation by at most 3.99e-5.
An independent NumPy fit/forward audit verifies all 22 exclusions and all 216
held-out vectors, with maximum scale-relative forward difference 1.43e-6.
It reconstructs the scalar metrics; it does not repeat the bridge's native
projection. All raw strings and 216 visual input calls are preserved, with
identical PNGs deduplicated to 172 files. No model weights are published.

Neither experiment admits an image-writing trial under its original rule.
The native visual-history route must also preserve invisible progress:
the separate [cursor-state audit](cursor_history_result.md) shows an actual
space-prefix collision and supplies a lossless visible-cursor interface.
That new format is not used by this probe or claimed to explain the preceding
glyph corruption. A learned reader, correct stopping and an autonomous
conditional image writer still need joint evidence on unseen inputs.

CMRC2018: Cui et al. (2019), [source](https://github.com/ymcui/cmrc2018),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Qwen3-VL: Alibaba, Apache-2.0. Copy requests are prepared examples.
