# More distinct sentences and the limits on visual translation

An independent two-million-parameter editor can learn a recurring classical Chinese construction from pictures and render a modern sentence with both copied and newly drawn glyphs. In a passage from the *History of Song*, it reads 十二月庚辰朔，日有食之。 and writes 十二月初一，日食。, matching the recorded modernization. In a passage from the *Kongzi Jiayu* it reads 汝止，吾将问之。 and writes 不不是，我们问他。 The reference asks the listener to wait while the speaker asks someone a question. The generated glyphs are legible, but the instruction and speaker are wrong. Figure 50 preserves both original pixel streams. [Figure 50](../publication/ilm-nature/figures/Figure_50_edit_diversity.png)

## Does greater coverage improve language at fixed exposures?

We compared four passes over 48,000 new paired sentences with one pass over 192,000 distinct pairs. The common checkpoint had already trained on another 48,000 pairs. Both arms saw 192,000 translation prompts and 48,000 quotation prompts over 1,920 optimizer steps. After the first shared block the parameter tensors had identical hashes. We reserved two fresh sentence sets from 789 chapters excluded from training. They share no chapters with one another. Earlier evaluations had used different sentences from 779 of these chapters. An initial attempt to exclude every previously evaluated chapter yielded only 85 discovery rows. We amended eligibility before fitting. Source strings and physical images were deduplicated across the new blocks and all preceding rosters. [Protocol](edit_diversity_protocol.md) · [Preparation receipt](evidence/edit_diversity_20261118/preparation.json)

| Training and decoder | Exact / 2,048, discovery | Glyph error | Exact / 2,048, replication | Glyph error |
|---|---:|---:|---:|---:|
| Four passes over 48,000 | 31 | 56.69% | 19 | 55.79% |
| One pass over 192,000 | 32 | 56.90% | 20 | 55.97% |
| Local edits, smaller corpus | 21 | 56.96% | 24 | 56.22% |
| Local edits, larger corpus | 25 | 56.65% | 23 | 55.83% |
| Larger corpus, relaxed insertion allowance | 23 | 68.61% | 18 | 67.87% |

Exact counts include all requests. Five discovery and six replication inputs or references could not be rendered. Glyph error divides total bitmap edit distance by renderable reference length. All three neural settings copied every renderable quotation exactly. In translation, just six of the larger-corpus editor's 52 exact successes required changing the source. Three of these changed answers exceeded eight glyphs, included shared and absent Han, and had complete target bitmaps absent from the arm's training targets. Alternative valid translations can score as errors against one recorded answer. The observed increase in glyph error for diverse training over repetition was 0.21 and 0.19 percentage points (95% paired chapter-bootstrap intervals 0.03–0.43 and 0.07–0.32). The larger corpus failed the registered improvement criterion. [Matched results](evidence/edit_diversity_20261118/comparison.json)

## Where the writer loses information

Given correct prior writing and the reference 48 physical ink measurements, the renderer exactly reconstructed 76,944 of 76,978 reference draws. With the same prior writing and the model choosing those measurements, it drew 8,569 correctly. Giving the editor more example sentences while freezing its source encoder and drawing motor changed neither endpoint much. These measurements distinguish knowing how to draw a shape from deciding which shape belongs in the sentence. They cover one printed typeface. [Discovery](evidence/edit_diversity_20261118/diverse_discovery_teacher.json) · [Replication](evidence/edit_diversity_20261118/diverse_replication_teacher.json)

For a source of length n and allowances b at its n+1 cursor positions, output length is bounded by U = min(192, n + sum(b)). Every copy or replacement consumes a source glyph. Each additional output costs one insertion. Even a perfect writer incurs at least max(0, reference length − U) edits. Summed over both cohorts, 44,211 of the 76,869 observed edits were unavoidable under the modal allowance. When the generated translation first diverged from the recorded edit alignment, the model copied instead of inserting in 2,100 cases. Other alignments can yield the same answer. [Capacity and first divergence](evidence/edit_diversity_20261118/capacity.json)

We fixed the conditional 90th percentile as an alternative upper allowance, kept every weight unchanged and generated the answers again. This reduced the number of unreachable lengths from 3,845 to 94, and the unavoidable edit count from 44,211 to 707. Actual errors rose from 76,869 to 92,944. Among the new policy's 55,109 independent drawings, only four failed to match a bitmap anywhere in the scoring bank. The extra output overwhelmingly consists of recognizable glyph shapes. Permitting more insertions fails as a remedy for choosing the proper phrase. Both tests used the cohorts that motivated this follow-up. [Allowance protocol](edit_diversity_allowance_protocol.md) · [Measured effect](evidence/edit_diversity_20261118/upper_allowance_comparison.json)

## What can be compressed?

Stacking glyph tiles in reading order preserves a sentence. A reshape to a two-dimensional grid with a known reading order is an invertible reindexing. Adding the slices together removes their order. For n glyphs with multiplicities m, up to n! / product(m!) sequences share a multiset and produce exactly the same summed ink. In the 192,000 physically distinct training inputs, 20 shared-multiset groups had different modern reference sequences. A deterministic function of summed ink must miss at least 24 recorded answers on this sample, even with unlimited training. One pair reverses the clauses 天下有兵 and 左将军死. The counting argument applies to commutative pooling. A learned representation that retains position can distinguish such pairs. The current question encoder also averages per-glyph features, so permutations of a prompt have the same representation in exact arithmetic. Its quotation/translation results test two distinct prompts, leaving ordered instruction following open. [Multiset counts](evidence/edit_diversity_20261118/capacity.json) · [Encoder](../ilm/visual_lm/visual_edit_writer.py)

The next experiment should train the distribution of complete inserted image spans. One candidate is a monotone transducer whose state is indexed by source position and the sequence of emitted glyphs. Such a state can sum over alternative placements of a correct span before taking the log likelihood. The current action-history GRU can reach different hidden states by different paths, so these states cannot simply be merged into an exact dynamic program. A comparison needs a history factorization that permits merging, a matched fixed-alignment control, held-out source-sensitive translations, and the actual cost of scoring the alignment lattice. This is a hypothesis about choosing the next phrase. Counterfactual prompts, other typefaces and open-ended book writing require separate tests. Related pixel-language work establishes prior art for autoregressive visual writing. [Graves, sequence transduction](https://arxiv.org/abs/1211.3711) · [Tai et al., PIXAR](https://aclanthology.org/2024.findings-acl.874/)

## Compute and replication

The additional fits used a single RTX 4090 D. Updating 242,901 of 2,026,237 parameters took 24.7 and 24.1 seconds and peaked at 419.8 and 421.2 MiB of allocated CUDA tensor memory on this host. The arms processed 7,647,364 and 7,664,255 supervised action positions. These are incremental fitting costs. They exclude parent training, data preparation and evaluation. Neither the present results nor the experiment's design establish a speedup over an LLM at matched language quality. [Repeated fit](evidence/edit_diversity_20261118/repeat_fit.json) · [Diverse fit](evidence/edit_diversity_20261118/diverse_fit.json)

From the repository root, after preparing the parent editor and earlier contrast rosters, run the following with the registered NiuTrans archive available.

```sh
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python scripts/prepare_edit_diversity.py
python scripts/compare_edit_diversity.py fit
python scripts/compare_edit_diversity.py evaluate
python scripts/audit_edit_diversity_capacity.py
python scripts/probe_edit_diversity_allowance.py
```

Each fitted checkpoint, frozen roster, raw output stream and score is stored under `artifacts/visual_edit_diversity_v1/`. To write a new answer from a stored fit, use the standard inference entry point.

```sh
python scripts/infer_visual_edit_translation.py \
  --checkpoint artifacts/visual_edit_diversity_v1/diverse/editor.safetensors \
  --source-image references/evidence/edit_diversity_20261118/success_source.png \
  --question-text 请翻译成白话文 --output artifacts/diversity_example
```
