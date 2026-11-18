# What local visual context teaches a Chinese phrase writer

Given the image of a classical Chinese sentence, a marked insertion boundary and the correct modern prefix, a small visual writer regenerated 405 of 5,987 held-out phrases exactly, including their ending. The preceding neural writer recovered 187. A rule that selects an observed whole phrase from local training counts recovered 651. The writer drew 我想 after 黄帝说： in a new sentence. In 爵禄者，兵之实也。 it drew 位的 after 爵 where the recorded translation continued 位俸. Both outputs stopped after two cells. [Original images and controls](../publication/ilm-nature/figures/Figure_53_local_ink.png)

| Exact stopped phrases | Discovery (5,376 sites) | Replication (5,413) | Fresh sentences (5,987) |
|---|---:|---:|---:|
| Previous neural writer | 337 | 268 | 187 |
| Local evidence for the first glyph | 599 | 538 | 354 |
| Local evidence throughout | 618 | 564 | 405 |
| Modern prefix alone, throughout | 323 | 328 | 268 |
| Whole-phrase training counts | 868 | 808 | 651 |
| Correct first drawing code supplied, then free | 2,439 | 2,232 | 1,917 |

The full writer improved on the original by 3.04–4.28 percentage points on fresh sentences (95% chapter bootstrap). It trailed whole-phrase counts by 3.32–4.89 points. It solved 104 sites the count rule missed and missed 350 the rule solved. Only 27 of 3,448 multi-glyph phrases and zero of 2,433 instances of unseen phrase types were exactly reproduced. It stopped on 5,188 sites. Six of the 1,024 fresh pairs could not be rendered. Their sources were deduplicated and their 549 chapters were outside training and both immediately preceding test pools. In 543 chapters other passages had appeared in older translation evaluations. [Fresh generation](evidence/local_ink_20261118/fresh_stream.json) · [Physical recount](evidence/local_ink_20261118/physical_recount.json) · [Roster](evidence/local_ink_20261118/fresh_roster.json)

## What the error measures tell us

**Drawing and phrase selection.** Of 35,290 fresh emitted glyphs, 35,279 matched a known bitmap in the evaluation typeface. The first emitted glyph matched the source translation at only 1,086/5,987 boundaries. Even granting the writer the correct stopping time at every site, its drawings could recover at most 590 fresh phrases, below the count rule's 651. The analogous ceilings on discovery and replication were 822 and 732. These are bounds for this fixed writer and decoding rule. Each replay reproduced the original free-generation score when the original ending decisions were restored. [Length audit](evidence/local_ink_20261118/fresh_length.json)

**Meaning in context.** A reader saw the classical sentence, the surrounding modern translation and anonymously ordered fillings for 48 sites from distinct fresh chapters. Half the reference insertions had multiple glyphs. A model reader involved in the study accepted 8 writer answers, 11 count-rule answers and 45 reference answers. Five accepted writer answers differed from the recorded translation. Four writer judgments were uncertain. The small stratified sample and the reader's involvement limit generalization. [Frozen reading decisions](evidence/local_ink_20261118/reading_result.json)

**How much information the local image loses.** Group the fresh sites by the same four physical input signatures. Any deterministic decision that retains only this context must give the same phrase to every member of a group. If the target counts in group $c$ are $n(c,y)$, its best possible exact score on these particular examples is $\sum_c \max_y n(c,y)$. This retrospective bound permits learning the answers from the test set. It measures the information lost by discarding wider context. There were 534 sites with conflicting targets for identical contexts. Even choosing the majority answer with hindsight loses at least 326 sites. Of those 534, the full neural writer solved 35, the prefix-only writer solved 10 and the count control solved 65. Most contexts (5,305 of 5,546) occurred just once, so the empirical conditional entropy of 0.156 bits per site has strong finite-sample downward bias as an estimate of the population entropy. The neural writer can also see the broader sentence. This test does not establish how well it exploits that information. [CPU recount without new model calls](evidence/local_ink_20261118/context_ambiguity.json)

## Model and costs

The same frozen 2,000,566-parameter writer receives a correction to the probabilities of twelve groups of four visible landmarks. For a group pattern $v$, group $r$, earlier drawing $h$ and local context $c$, the normalized update is

$$
q_r(v\mid h,c) =
\frac{p_{\theta,r}(v\mid h,c)\,[\hat p_r(v\mid c)/\hat p_r(v)]^{w_r}}
{\sum_{u}p_{\theta,r}(u\mid h,c)\,[\hat p_r(u\mid c)/\hat p_r(u)]^{w_r}}.
$$

This is an additive change to the negative log probability, the statistical-mechanics meaning of an energy. Normalization is over sixteen physical patterns. A common pattern receives extra weight only when it is more frequent in this context than globally. The local correction sees already emitted modern glyphs at inference. Its count tables contain subpatterns and binary ending observations. Each completed drawing becomes physical input for the next one. The empirical correction ignores correlations between groups inside a glyph, while the neural component conditions on earlier drawn groups. Conditional image densities and autoregressive text generation in pixel space both have prior demonstrations. [Pixel recurrent neural networks](https://proceedings.mlr.press/v48/oord16.html) · [PIXAR](https://aclanthology.org/2024.findings-acl.874/)

Training used 64,384 parallel sentences, 296,837 first glyphs, 330,601 later glyphs and 627,438 post-emission ending decisions. On the 4090 D workstation the two conditioned fitting and calibration stages took 6.8 and 10.5 seconds, excluding their respective shared preparation (5.0 and 4.1 seconds), previous neural training and evaluation. Fresh generation took 17.7 seconds in batches with 114.6 MiB peak allocated GPU memory. The three extra count arrays occupy 76.5 MB unpacked and 18.95 MB compressed; model weights, framework and working memory are additional. A matched-quality comparison against a text language model remains open. [Onset fit](evidence/local_ink_20261118/onset_source_fit.json) · [Subsequent cells and endings](evidence/local_ink_20261118/stream_source_fit.json) · [Published table sizes](evidence/local_ink_20261118/public_manifest.json)

## Reproduction and next comparison

The released model runs from PNG inputs on CPU. The training programs use the established private corpus preparation and frozen neural checkpoints. Original fit and generation programs create fresh output directories. In a fresh training workspace the stages run in this order, with the repository root on the Python path. [Runnable model and CPU example](../models/visual_edit_zh/README.md) · [Original training prerequisites](visual_phrase_protocol.md)

```sh
export PYTHONPATH=.
python scripts/train_local_ink.py
python scripts/evaluate_local_ink.py
python scripts/prepare_local_ink_holdout.py
python scripts/train_stream_ink.py
python scripts/evaluate_stream_ink.py
python scripts/audit_local_ink_length.py
python scripts/audit_local_ink_context.py
python scripts/export_local_ink.py --output /tmp/ilm-local-ink-tables
```

The candidate failed the preregistered whole-phrase, novelty and ending requirements for sentence integration. The next bounded experiment is to condition the local drawing correction on one earlier four-pixel group in the **same** glyph. This tests whether a joint visual pattern helps choose the intended glyph. Freeze the sentence reader, training data, stopping rule and decoding budget. Select backoff and weight on training chapters only. Require an improvement over the present writer and the count control on unseen sentences, a correct unseen phrase, at least 99% stopping in each cohort, and evaluate same-neighborhood, different-target cases and contextual reading separately. If held-out likelihood improves without better phrase choice, discontinue this family of local corrections and test supervision that distinguishes whole-sentence meanings. The existing evidence leaves general visual language and the proposed thousandfold resource advantage unresolved. [First-glyph protocol](local_ink_protocol.md) · [Stream protocol](local_ink_stream_protocol.md) · [Length audit design](local_ink_length_protocol.md)
