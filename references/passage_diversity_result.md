# More pictured passages at the same training budget

The Chinese writer drew 风云变幻 for the passage “富国基金在〔　〕的市场中彰显着她的实力与魅力”. In a passage about solving a football club's remaining problems it drew 不得而知 where the source used 迎刃而解. These are the first gain and loss in source order against the writer trained on repetitions. Actual input windows and independently produced bitmaps appear in the figure. [Figure 44](../publication/ilm-nature/figures/Figure_44_passage_diversity.png)

| Training condition | Fresh exact answers / 17,862 | Both correct / 6,772 pairs | Distinct outputs | Ten most frequent outputs |
|---|---:|---:|---:|---:|
| Released starting writer | 410 | 5 | 372 | 54.8% |
| Repeated examples | 516 | 5 | 549 | 35.6% |
| Additional distinct passages | 550 | 7 | 369 | 41.1% |

Both new fits saw 571,512 windows and made 2,980 updates from the same checkpoint on one RTX 4090 D. Doubling the number of distinct windows to 142,878 gave 34 additional exact expressions relative to repetition. The paired difference was 0.19 percentage points (95% source-group interval 0.00–0.39). The preceding 17,555 gaps yielded 523 versus 513. The local count baselines recovered 972 and 778 fresh expressions from the combined and original corpora. The diverse fit's accuracy advantage over repetition remained uncertain and its top-ten output share was higher. Only four fresh windows contained target glyphs that the renderer failed to reproduce from their canonical measurements. [Registered comparison](passage_diversity_protocol.md) · [Data and exclusions](evidence/passage_diversity_20261118/data.json) · [Fits and renderer coverage](evidence/passage_diversity_20261118/fits.json) · [Scores](evidence/passage_diversity_20261118/evaluation.json)

A second experiment changed the contribution of each recorded expression to the training loss. With inverse-square-root frequency weighting, the model recovered two of 2,161 cases with one to twenty training occurrences, versus zero with uniform weights. Overall it fell to 427/17,962 exact answers from 521. It gained 69 and lost 163 (paired difference −0.52 percentage points, 95% interval −0.68 to −0.35). The fraction occupied by the ten most frequent generated answers rose from 41.0% to 52.5%. Neither follow-up reached the registered release criteria. The current independent checkpoint is the previously confirmed joint reference writer. [Frequency-weighting protocol](expression_frequency_weight_protocol.md) · [Held-out measurements](evidence/passage_diversity_20261118/frequency_weight.json) · [Runnable model](../models/pictured_idiom_zh/README.md)

The local baseline and the uniform-weight writer solved different passages. Of 974 correct baseline answers, 739 were missed by the writer. The writer alone solved 286. In 194 cases all eight glyphs nearest the gap matched a training window. The local baseline solved 144 and the writer solved 16, with fourteen shared successes. These 194 cases directly motivate testing whether an ordered group of pictured glyphs can provide a more useful learned input to the writer. Repeated local phrases account for part of this diagnostic. Whole-passage performance on new source groups remains the criterion for any resulting model. [Saved-output error analysis](evidence/passage_diversity_20261118/local_errors.json)

## Reproducing the comparison

With the preceding ChID data and joint-reference artifacts prepared, run these modules from the project root. Each comparison saves complete model-generated pixels and stops in its own artifact directory. The intentionally separate execution phases preserve the checkpoints if evaluation is interrupted.

```sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m scripts.compare_passage_diversity prepare
python -m scripts.compare_passage_diversity train
python -m scripts.compare_passage_diversity evaluate
python -m scripts.test_expression_frequency_weight
python -m scripts.measure_visual_neighborhood_errors
```

Preparing the original comparison took 19.7 s. The two fits and their setup took 58.0 s, with 280 MiB peak allocated device memory, and evaluation took 45.1 s. The subsequent weighted fit took 29.8 s within a 64.9 s preparation, training and evaluation run. These incremental costs exclude the original renderer, reader and writer training. The first evaluation launch tripped the process RSS check before inference. A fresh child process passed the same limit and evaluated the saved checkpoints. [Preparation](evidence/passage_diversity_20261118/prepare_resources.json) · [Training](evidence/passage_diversity_20261118/train_resources.json) · [Evaluation](evidence/passage_diversity_20261118/evaluate_resources.json)
