# Writing a missing Chinese idiom from pictured prose

A 2,115,971-parameter reader and writer produced the exact missing expression for 134 of 22,907 Chinese passages. In 100 of those successes, all four answer characters were absent from the visible window. The model reads two cropped text images, draws its answer one cell at a time and decides when to stop. The checkpoint runs independently on a CPU. [Run the checkpoint](../models/pictured_idiom_zh/README.md) · [See the original output images in Figure 40](../publication/ilm-nature/figures/Figure_40_pictured_idioms.png)

The first exact result in validation order with all four glyphs missing was **不寒而栗** in the passage “……这是令人想起来都〔　〕的。实际上无论你抗议也罢……”. The first error was **一波三折** for “篮协改革联赛切莫〔　〕”, where the source had **因噎废食**. A fresh process given typed context around a different gap drew **轻举妄动** byte for byte. The input renderer supplied only the surrounding passage. Each result below requires the complete reference bitmap sequence and a stop at the correct length. A different plausible expression receives zero credit.

## Training and evaluation

The pictured corpus comes from ChID. We removed the answer choices, retained up to 24 glyphs on each side of each gap, and rendered a single neutral marker at every other gap. Hash sampling by original source group selected 71,439 eligible training windows in 10,529 groups with 3,805 answer types. The official validation split supplied 22,907 eligible windows in 3,218 groups, with 3,410 answer types. Training and validation share no complete pictured windows. Of 22,848 validation windows whose central 16 cells were new, 133 were solved by the aligned model. The official test split was left unopened. The source benchmark uses a different, multiple-choice task. [ChID paper](https://aclanthology.org/P19-1075/) · [Pinned source distribution](https://huggingface.co/datasets/clue/clue/tree/28178267a609dd08bdc703dd6c931dfc2c2f4431/chid) · [Preparation counts](evidence/pictured_idioms_20261118/preparation.json)

| Method | Complete exact | Exchanged context | Distinct correct expressions containing absent Han |
|---|---:|---:|---:|
| Predict successive groups of four physical bits | 35 | 15 | 4 |
| Add whole-answer alignment during training | 134 | 23 | 43 |
| Choose among 33 self-generated answers | 143 | 10 | 58 |

The added training loss rewards the reader when its passage representation is close to the image representation of the complete answer. Repeated identical answers in the same minibatch are all positives. The drawing head, training order and deployed architecture were held fixed. Against the original writer, 125 formerly wrong answers became exact and 26 exact answers were lost. Resampling the source groups gave a 95% interval of +0.32 to +0.55 percentage points. Of the 16,797 passages missing every target character, 100 were completed correctly. The 38 passages whose entire target expression was unseen in training yielded zero exact completions. Of 8,900 disjoint pairs sharing both adjacent glyphs but having different references, the writer changed its answer on 6,846 and solved both in zero. These pairs and novelty counts bound what this experiment can say about wider context. [Training protocol](pictured_idiom_alignment_protocol.md) · [Alignment results](evidence/pictured_idiom_alignment_20261118/result.json) · [Independent recount](evidence/pictured_idiom_alignment_20261118/recount.json) · [Pair construction](pictured_idiom_context_pairs.md)

## Where the language signal is lost

The input encoder reads regional ink coverage and physical pixel measurements in order. A recurrent head emits groups of four binary measurements and receives feedback from the completed glyph. Forty-eight chosen pixel coordinates distinguish all 2,323 training target shapes. A learned renderer reconstructs each 32 × 32 cell from these measurements. Given the true measurements, it reconstructed 2,229 of 2,229 training-known validation shapes and zero of eight unseen shapes. Of 91,512 freely emitted cells, 91,474 exactly matched *some* training glyph. These numbers establish a compact representation of familiar printed shapes. Generalization to new glyphs, handwriting and typefaces needs a separate test. [Model](../ilm/visual_lm/idiom_ink.py) · [Physical training protocol](pictured_idiom_protocol.md)

A count model that backs off from eight local glyphs to smaller neighbourhoods recovered **1,012/22,907** expressions. Given all 3,805 pictured training answers, the trained reader chose **930/22,907** correctly. Of those 930, the independent writer solved only 79. It also solved 55 that the supplied-answer reader missed and 70 that the local baseline missed. The 851 reader successes lost in generation give a concrete target for improving the writer. The reader itself still has substantial room to improve. Comparing bare inference times between these systems would be misleading at their present accuracies. [Same-passage audit](evidence/pictured_idiom_alignment_20261118/information.json)

Reranking 32 extra samples plus the original beam gave 143 exact answers. Its improvement over the aligned writer had a 95% interval of −0.017 to +0.100 percentage points. Even a selector with perfect reference knowledge could recover only 175 passages from those 33 proposals. The additional draws cost 51.5 seconds across validation and 34,872 of the selected cells matched no training target shape. The simpler aligned writer remains the published checkpoint. [Proposal protocol](pictured_idiom_generated_choice_protocol.md) · [Recount of all 755,931 proposals](evidence/generated_idiom_choice_20261118/proposal_audit.json)

## What one image can preserve

For glyph layers numbered 0 to 3, summing their masks gives the same integer image for every permutation. Across the 3,410 validation expressions there were 3,376 distinct sum images. In 34 groups, different attested expressions literally became the same image. Even assigning the most frequent reference to each sum leaves at least 115 errors on this fixed corpus if only the sum is available. Reversing every stream changed the ordered reference in all 22,907 cases while preserving its sum.

Packing the four bits in reading order into one four-bit value per pixel recovered every ordered stream exactly. This uses the same information capacity as the original four binary planes. A spatial tiling with a known reading order is another reversible layout. Either preserves a distinction a language learner may need. Neither produces information about which expression belongs in a new passage. PIXEL and PIXAR already establish precedents for reading rendered text and generating text images. The specific issue measured here is transfer from recognizing an expression in context to writing it independently. [Physical audit](../scripts/audit_idiom_information.py) · [Shannon](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) · [PIXEL](https://arxiv.org/abs/2207.06991) · [PIXAR](https://aclanthology.org/2024.findings-acl.874/)

## Reproduction

From the repository root with the project training environment active, the following commands fetch the two pinned splits, prepare their images and train both controls. Run the fits in a fresh workspace for fresh output directories. The official validation references are used only for scoring. The inference command in the model card can run directly from the supplied checkpoint.

```bash
PYTHONPATH=. python -m scripts.fetch_pictured_idiom_sources
PYTHONPATH=. python -m scripts.prepare_pictured_idioms
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m scripts.fit_pictured_idioms
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m scripts.fit_pictured_idiom_alignment
PYTHONPATH=. python -m scripts.audit_idiom_information
```

Pictured-corpus preparation took 11.5 seconds. On one RTX 4090 D, predictive fitting took 46.3 seconds for the original writer and 57.2 seconds with alignment. Their fit-and-evaluation runs took 56.4 and 68.2 seconds. The aligned run reused the original trained renderer and used 289 MiB peak allocated device memory. It used 246,336 auxiliary training parameters beyond the published checkpoint. These measurements cover this cloze experiment. General dialogue, an illustrated etymology of 言 and computational savings at equivalent language quality remain open. [Standalone model and inference](../models/pictured_idiom_zh/README.md) · [Manuscript](../publication/ilm-nature/manuscript.pdf)
