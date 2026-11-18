# Reading a classical sentence and writing its modern form

A two-million-parameter editor trained on a single RTX 4090 D produced complete, previously unseen modern sentences by mixing copied source ink with its own drawings. On the second set of chapters it exactly reproduced three of 1,964 long references that needed a Han glyph absent from the source. Its aggregate edit distance still failed the training-only local-edit comparison. [Figure 48](../publication/ilm-nature/figures/Figure_48_visual_edit.png) · [Runnable checkpoint](../models/visual_edit_zh/README.md)

## What the model wrote

The table includes **every** exact longer mixed output on the separate confirmation chapters for the learned insertion limit. The complete target sequences were absent from the training roster. In the second row the reference date conversion is specific to the recorded context. Exact reproduction of that date does not establish calendar reasoning.

| Pictured source, transcribed here | Model output, equal to the reference pixels | Copied | Drawn |
|---|---|---:|---:|
| 四年，改为侍中、金紫光禄大夫。 | 四年，改任侍中、金紫光禄大夫。 | 14 | 1 |
| 二年春正月甲寅，车驾还宫。 | 二年春正月初二，车驾回宫。 | 10 | 3 |
| 秋七月，太尉穆崇薨。 | 秋七月，太尉穆崇去世。 | 9 | 2 |

The first failure on the first roster illustrates the scientific limit. For 丁亥，诏停蔡州乡军。 the writer produced 十二日，下诏停蔡州乡军。 The reference begins 二日 and expands 停 to 取消. Both outputs contain readable writing, with a consequential difference in date. Every example was scored from the saved packed bitmaps and stop action. [Raw failure pixels](evidence/visual_edit_20261118/error_actual.png) · [Independent recount](evidence/visual_edit_20261118/recount.json)

## Sentences, controls and costs

NiuTrans aligned the classical sources and modern translations automatically. A chapter-hash split preceded selection. We trained on 48,000 unique pictured sources from 4,882 chapters with 1,563,174 target glyph cells. The first 2,048 requests came from 254 chapters, and the second 2,048 from a separate 248. Four and three requests respectively exceeded input/target raster coverage. Exact counts use the full request denominator. Edit rates use the renderable references and can exceed 100% if the system keeps inserting. The local control predicts edits from up to three training-source neighbours. [Corpus and alignment](https://github.com/NiuTrans/Classical-Modern) · [Preparation receipt](evidence/visual_edit_20261118/preparation.json)

| Method | Exact translations, first / confirmation | Edit rate, first / confirmation | Longer mixed, first / confirmation |
|---|---:|---:|---:|
| Copy the source | 41 / 37 | 56.45% / 57.28% | 0 / 0 |
| Frequent local edits | 36 / 29 | 55.48% / 56.39% | 2 / 2 |
| Initial editor | 30 / — | 372.13% / — | 0 / — |
| Repaired renderer | 30 / — | 371.20% / — | 0 / — |
| Fixed five-insertion limit | 30 / — | 79.82% / — | 0 / — |
| Learned insertion limit | 32 / 30 | 55.93% / 56.82% | 1 / 3 |
| Joint action and drawing score | 42 / 38 | 60.23% / 62.48% | 1 / 1 |
| Joint score and learned limit | 42 / 38 | 55.57% / 56.36% | 1 / 1 |

All 1,291 initial first-roster capped translations ended with at least sixteen insertions at one source cursor. Learning a position-specific limit eliminated capped responses on both rosters. The learned-limit writer also reproduced all 2,044 and 2,045 renderable quotations. Exchanging length-neighbouring source sentences left zero exact translations on either roster. Known phrasings were used. Translation exactness measures physical agreement with one recorded rendering. Other phrasings may be valid and were not human-graded. The paired chapter-bootstrap 95% intervals for improvement of the joint score and learned limit over local counts were [−0.38, 0.22] and [−0.39, 0.45] percentage points. Every tested editor missed the specified translation qualification. [First-roster comparison](evidence/visual_edit_20261118/joint_choice_joint_limited_heldout.json) · [Confirmation](evidence/visual_edit_20261118/joint_choice_joint_limited_confirmation.json)

The initial 1,984,053-parameter model used 568,373 trainable editor parameters after motor fitting. Eight epochs provided 384,000 translation and 96,000 quotation presentations. The initial fit took 38.55 seconds with 468.85 MiB peak allocated GPU memory. The renderer correction took 8.00 seconds and the 42,184-parameter insertion head 9.61 seconds. These are measured training phases on one RTX 4090 D. Corpus preparation, scoring and all other project experiments incur additional costs. A matched-quality comparison with an 8B model remains open. [Training](evidence/visual_edit_20261118/fit.json) · [Renderer](evidence/visual_edit_20261118/motor_refined.json) · [Insertion count](evidence/visual_edit_20261118/fertility.json)

## What the interventions identify

Let a pictured source and instruction determine a state $h$ with the previous actions and emitted pixels. A draw operation chooses twelve groups of four measured pixels. For a proposed trajectory with action $a$ and groups $b$, its local probability is $p(a\mid h)\prod_{j=1}^{12}p(b_j\mid h,b_{<j})$. Adding the log probability of the best retained pattern distinguishes the probability of a specific drawing from the probability of drawing anything. The within-cell beam has width four. Several trajectories can yield the same image, and local mode selection need not maximize the probability of a complete sentence.

Supplying the correct previous writing and cursor removes accumulated rollout error. At the 37,451 recorded draw positions, the unchanged language state predicted exactly the target picture 3,816 times (10.19%). Giving the same renderer the correct landmark measurements produced 37,427 exact pictures (99.94%). The four-proposal beam and single forced alignment still limit this diagnostic. The gap shows where the reference information was lost on this test. Further perfecting the motor alone leaves that information unavailable. [Supplied-history measurements](evidence/visual_edit_20261118/joint_choice_teacher_history.json)

### Can a stack of glyphs be flattened?

Packing an ordered stack into known grid slots is reversible. Summing it through depth loses order. For two distinct images $A$ and $B$, $A+B=B+A$. Even a noiseless sensor with an exact sum and the stream length cannot distinguish the two orders. Under a uniform prior over the two inputs, expected exact recovery is at most one half. Providing position or separate channels can restore identifiability. The number of possible bitmaps per cell measures representational capacity, while the conditional distribution of meaningful continuations determines how much the model has to learn.

We tested the sum on the actual 48,000 training passages. Three pairs had identical lengths and exact 1,024-pixel sums, including 时年六十五。 and 时年五十六。 (ages 65 and 56). Each pair had distinct reference translations. Any deterministic reader restricted to that measurement must miss at least three exact copies and three exact translations in this roster. This empirical lower bound says little about the other 47,994 examples or the general language error rate. [Order measurement and pair indices](evidence/visual_edit_20261118/order_loss.json)

### A falsifiable next learning comparison

Retain an ordered input and the working motor. Compare the existing pattern loss with the same loss plus one training-only objective that contrasts the correct complete drawing against other observed drawings in the minibatch. Score a drawing by its existing conditional pattern log probability. Exclude duplicates of the positive bitmap. At inference the writer must generate its own measurements and pixels. Keep distinct chapter evaluation, compute, optimizer updates and examples matched, include a fresh untouched chapter set, and preserve the local and exchanged-source controls. Inspect the supplied-history drawing accuracy **and** independently produced stopped answers. A gain limited to supplied-history selection does not qualify the writer. If the trained context still fails to outperform local edits on fresh chapters, test whether the encoder discards phrase order with matched order-changing inputs before scaling it.

Pixel-space language generation has experimental precedents. PIXAR measures generative language tasks and legibility. Chu and colleagues report a 33M pixel model solving controlled visual arithmetic, counting and drawing tasks with intermediate frames. Contrastive predictive coding provides a training objective for distinguishing a true future from alternatives. These results motivate task-specific tests of context. The present measurements establish neither broad semantic competence nor a compute multiplier. [PIXAR](https://aclanthology.org/2024.findings-acl.874/) · [Pixel is All You Need, preprint](https://arxiv.org/html/2611.22820v1) · [Contrastive predictive coding](https://arxiv.org/abs/1807.03748)

## Reproducing the training and the physical audit

The research checkpoint supports immediate CPU inference. To reproduce training, use a fresh checkout with the repository dependencies and Noto Sans CJK installed. The preparation and evaluation programs reserve new output directories and verify inputs. Use the pinned archive associated with the receipt. The commands below start in the repository root.

```sh
export PYTHONPATH=.
mkdir -p data/external/classical_modern_v1 artifacts
curl --fail --location 'https://codeload.github.com/NiuTrans/Classical-Modern/tar.gz/4e746ea9fa99c3c0d7051c45397330bef7b0962d' -o data/external/classical_modern_v1/archive.tar.gz
python scripts/prepare_visual_edit_translation.py
python scripts/train_visual_edit_translation.py
python scripts/evaluate_visual_edit_translation.py
python scripts/refine_visual_edit_motor.py
python scripts/evaluate_visual_edit_translation.py --checkpoint artifacts/visual_edit_translation_v1/motor_refined/editor.safetensors --destination artifacts/visual_edit_translation_v1/evaluation_refined
python scripts/fit_visual_edit_fertility.py
python scripts/evaluate_visual_edit_counts.py
python scripts/evaluate_visual_edit_joint.py
python scripts/recount_visual_edit_translation.py artifacts/visual_edit_translation_v1 /tmp/ilm-editor-recount.json
python scripts/measure_visual_order_loss.py artifacts/visual_edit_translation_v1 /tmp/ilm-order-loss.json
```
