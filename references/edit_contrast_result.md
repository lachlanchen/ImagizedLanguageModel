# Choosing a pictured glyph from its sentence

An independently trained visual editor used the order of glyphs in classical Chinese to predict modern writing. With its previous writing and source cursor supplied, it drew 5,633 of 55,758 requested glyphs exactly. Reversing the source while preserving the actual glyph under its cursor reduced this to 4,695. Training the same editor to distinguish a complete target glyph from seven other drawings improved neither its translations nor its glyph fidelity. [Figure 49](../publication/ilm-nature/figures/Figure_49_edit_contrast.png)

## Three matched fits

All three fits began from the preceding two-million-parameter editor. Each updated the same 242,901 decoder parameters over the same 1,920 batches, including 192,000 presentations of translations and 48,000 presentations of quotations. The input encoders, output renderer and insertion allowances were frozen. Both initial comparison arms computed the whole-glyph scores. Only the contrastive arm applied their gradients. The subsequent corrected arm had the same sample schedule. [Training protocol](visual_edit_contrast_protocol.md) · [Correction protocol](edit_noise_correction_protocol.md)

Let $s(b,h)$ be the log probability of the twelve predicted ink groups of drawing $b$ given history $h$. The added training objective scores the correct complete drawing among eight pictures using a softmax over $s$. With an independent noise distribution $q$, Bayes' rule gives a posterior over the target index proportional to $p(b\mid h)/q(b)$. Thus the score that solves the discrimination task can differ from the log probability required for free generation. The corrected arm used $s(b,h)-\log q(b)$ for comparison and retained $s(b,h)$ for generation. Its marginal estimate used 866,590 training draw positions and 5,115 distinct codes. Minibatch candidates were sampled without replacement inside length buckets, so the independent-noise derivation is an approximation here. [Contrastive predictive coding](https://arxiv.org/abs/1807.03748) · [Correction measurements](evidence/edit_contrast_20261118/correction.json)

## Complete modern translations

Every cell in the table gives **exact stopped sentences / glyph edit rate**. Both rosters contained 2,048 requests. The 1,020 and 755 chapters were disjoint from one another and from earlier training and evaluation. Five and eleven requests respectively exceeded font coverage. They count against whole-answer accuracy. Glyph error rates divide total physical edit distance by total renderable reference length. The same likelihood-trained control and the corrected model were scored on the further roster, which was reserved before training the correction. The original held-out roster also remains available in the evidence directory.

| Method | 1,020 fresh chapters | 755 further chapters |
|---|---:|---:|
| Copy source | 6 / 69.19% | 12 / 70.55% |
| Local edit counts | 6 / 67.91% | 10 / 69.23% |
| Initial editor | 3 / 68.35% | — |
| Extra likelihood training | 4 / 67.90% | 7 / 69.23% |
| Whole-glyph contrast | 2 / 69.92% | — |
| Frequency-corrected contrast | 2 / 68.81% | 7 / 70.25% |

All evaluated models copied every renderable passage when asked to quote. Exchanging the source sentences left zero exact translations. On the first roster the likelihood control produced two exact answers that changed the source, including 何谓气？ → 什么是气？. The two contrastive models' only exact answers were identical to the source. The uncorrected objective increased the edit rate by 2.02 percentage points relative to equal likelihood training (paired 95% chapter-bootstrap interval 1.75–2.29). After frequency correction the increase was 0.91 points (0.71–1.12) and 1.02 points (0.85–1.20) on the two rosters. These are single-seed paired comparisons. A reference mismatch can include a valid alternative translation. [Independent pixel and action recount](evidence/edit_contrast_20261118/correction_recount.json) · [Matched comparison](evidence/edit_contrast_20261118/comparison.json)

## Why the pictures deteriorated

With correct prior history, the likelihood, contrastive and corrected fits drew 5,633, 1,546 and 2,796 reference glyphs respectively in 55,758 opportunities. All used the same frozen renderer, which produced 55,736 exact glyphs from the true measurements. During independent generation, just **2 of 8,025** newly drawn cells from the likelihood control matched no bitmap in the scoring collection. The corresponding counts were **6,982 of 8,760** for the contrastive model and **4,762 of 8,523** after correction. Copied cells were excluded. A new image can still be readable and valid, particularly in a different typeface. The actual outputs in this single-font experiment visibly lose strokes, including in both examples shown in Figure 49. Training only against other observed writing leaves other combinations of ink measurements outside the sampled comparison. The experiment measures that failure and the partial effect of frequency correction. It cannot assign every error to a single cause.

Natural matched pairs gave a second view of what the model learned. The two members came from different chapters and had the same current source bitmap, most recently emitted glyph, current edit action and previous action, with different next reference drawings. Among 6,847 pairs, likelihood training assigned the two pictures correctly 3,963 times (57.9%) and drew both correctly in 15 pairs. The uncorrected contrast gave 3,985 assignments and one pair with both drawings correct. Source reversal cost the likelihood control 1.68 percentage points of supplied-history drawing accuracy (95% interval 1.49–1.88). These interventions show that surrounding ordered writing affects prediction. They do not establish long-range semantic inference. [Order protocol](edit_contrast_order_protocol.md) · [Natural-pair and order results](evidence/edit_contrast_20261118/likelihood_source_order.json)

## Cost and decision

Each fit ran on one RTX 4090 D. The three additional fits took 17.82, 15.53 and 16.15 seconds respectively with at most 423 MiB allocated GPU memory during fitting. These figures exclude original training, preprocessing and evaluation and cannot establish an efficiency advantage at matched language quality. No candidate passed the release criteria. The public checkpoint continues to reproduce the earlier measured behavior. [Runnable editor](../models/visual_edit_zh/README.md)

The next comparison should retain likelihood and stable independent drawing, and change the number of distinct real translation examples at matched exposure. The current four extra passes revisit 48,000 sentences. A data-diversity test can ask whether additional, independent examples produce more complete changed sentences and beat the local editor on untouched chapters. A training-only selection score has little value unless its gain survives free writing.

## Reproduction

After preparing the original sentence corpus and obtaining the preceding research checkpoint, run the following commands from the repository root. The comparison writes new files under `artifacts/visual_edit_contrast_v1`. Evaluation preserves original predictions, actions, errors and checkpoint hashes.

```sh
export PYTHONPATH=.
python scripts/prepare_edit_contrast_chapters.py
python scripts/compare_visual_edit_contrast.py fit
python scripts/compare_visual_edit_contrast.py evaluate
python scripts/probe_visual_edit_order.py
python scripts/audit_edit_contrast_streams.py
python scripts/prepare_edit_contrast_chapters.py --confirmation
python scripts/compare_visual_noise_correction.py
python scripts/audit_edit_contrast_streams.py --corrected
```
