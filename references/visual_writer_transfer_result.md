# Writing an expression recognized by a pictured reader

A 2.12-million-parameter visual model completes a missing four-character expression from the pictured passage around it. For “MBA在美国经历了百年的〔　〕，有着很强的生命力”, the new writer drew **风风雨雨**. It also drew **不可告人** for a passage about a price change whose reference was **风吹草动**. Both examples are free outputs from the trained model. [Figure 41](../publication/ilm-nature/figures/Figure_41_writer_transfer.png) · [Run the checkpoint](../models/pictured_idiom_zh/README.md)

| Method | Exact on 22,907 development gaps | Exact among 930 reader successes | Exact on 9,135 new gaps | Exact among 398 reader successes |
|---|---:|---:|---:|---:|
| Initial aligned writer | 134 | 79 | 60 | 44 |
| Same writer, continued on original targets | 292 | 169 | 125 | 79 |
| Same writer, continued with reader guidance | 378 | 263 | 163 | 112 |
| Local count baseline | 1,012 | | 406 | |

Both continued writers saw the same 71,439 training passages over 24 additional epochs. They share an identical reader, renderer, trainable architecture, initialization, training order, inference budget and stopping rule. For the guided writer, the frozen reader scored 3,805 training expression pictures. On each visit to a training passage the writer received either the original response or a complete expression drawn from that distribution. This changed 1,212,654 of 1,714,536 targets. Once trained, the writer runs on CPU from the original weights and a 1.1 MiB file of updated recurrent and output weights. Correctness requires matching every output pixel and stopping at the right length. Other sensible expressions receive zero credit. [Training protocol](visual_writer_distillation_protocol.md) · [Sequence-level knowledge distillation](https://aclanthology.org/D16-1139/) · [ChID](https://aclanthology.org/P19-1075/)

**What transferred.** The guided writer gained 296 development answers and lost 52 relative to the initial model. It recovered 190 of the 851 answers that the reader already recognized but the initial writer missed. It also solved 115 that the reader missed. The gain over the matched continued writer was 0.38 percentage points (95% source-group interval +0.21 to +0.54). Of 16,797 windows in which all four target glyphs were absent, it completed 268. Of 38 target expressions absent from training, it completed zero. Its 378 exact answers cover 106 distinct expressions. The learned renderer reproduced a training-known shape at 91,235 of 91,622 emitted positions. These scores concern completion of pictured modern Chinese prose. [Full comparisons](evidence/writer_transfer_20261118/result.json)

**Fresh passages.** We set aside 1,356 additional hash-selected groups from the original ChID training source, disjoint from the groups used to fit the model. We removed 55 duplicate or overlapping windows and 16 unsupported renderings. On these 9,135 examples, guided training gained 0.42 percentage points over matched continued training (95% interval +0.19 to +0.65). It solved 111 of 6,676 all-four-absent cases and zero of 18 unseen whole expressions. Among fixed pairs sharing the same two adjacent glyphs and requiring different answers, it solved both members of 2/3,237 fresh pairs and 0/8,900 development pairs. The official test split remains unopened. [Fresh group definition](visual_writer_fresh_groups_protocol.md)

**Near and distant context.** With the nearest eight cells alone, the guided writer completed 126 fresh answers exactly, losing 66 intact successes and gaining 29. With the outer cells alone it completed seven. Both fresh pairs lost at least one answer under either mask. The surrounding passage contributed to some answers, while the remaining counts show considerable reliance on the words next to the gap. Padding portions of a natural passage changes its input distribution. [Context intervention](visual_writer_context_distance_protocol.md) · [Per-condition counts and pairs](evidence/writer_transfer_20261118/context_distance.json)

**Cost.** Preparing the teacher took 0.39 seconds. The matched and guided fits took 43.0 and 36.6 seconds. Both fits, final development evaluations and common preparation together took 98.1 seconds and peaked at 335 MiB of allocated GPU memory on one RTX 4090 D. The fresh evaluation, including rendering and three writers, took 22.2 seconds. Earlier reader and renderer training have their own cost. A cold CPU run reproduced the first new correct output byte for byte. Neither a 1,000-fold compute advantage nor broad language understanding has been established. [Earlier reader and renderer](pictured_idiom_result.md)

To reproduce the comparison, first train the aligned writer as described in the earlier report. From a fresh output workspace run the commands below. The trained model can also be used directly with the published weights.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m scripts.fit_visual_writer_distillation
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m scripts.confirm_visual_writer_groups
PYTHONPATH=. python -m scripts.score_visual_writer_transfer --output artifacts/transfer_report
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m scripts.measure_visual_writer_context --output artifacts/transfer_report
```

The scoring and masking commands use the same fresh report directory. All physical predictions and both final fitted states are retained in the experiment output.
