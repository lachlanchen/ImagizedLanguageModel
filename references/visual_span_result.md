# Learning which phrase belongs in pictured Chinese

A two-million-parameter writer learned that the order of classical Chinese glyphs changes which modern phrase fits. When two possible inserted phrases were offered, the writer preferred their recorded assignment in 213 of 293 natural sentence pairs. Reversing the classical sources reduced this to 161. All exchanged glyphs were absent from both sources. Given only the source image and a translation prompt, the most probable next-image decoder returned zero exact changed translations on the same two cohorts. [Figure 51](../publication/ilm-nature/figures/Figure_51_visual_span.png)

## Training on a complete answer image

A modern sentence can be reached by several alignments of copying, drawing and deleting. To sum them exactly, the recurrent history must agree wherever two paths have produced the same observed glyph codes at the same source position. We used the ordered source representation and a recurrent state updated only on emission. Each observed output supplies 48 ink measurements. A 32-by-32 renderer restores a full glyph and writes its 48 measured bits back to their physical locations. Copying can return any visible source image. New glyphs are drawn by a learned distribution over twelve successive four-pixel patterns. At deployment the writer consumes its own images and uses no Unicode output classifier or external teacher. The current prompt encoder averages individual pictured glyph features, which limits general instruction following.

We trained two instances from the same visual editor. Both scored the same source–target lattice. One optimized the probability of a recorded minimum-edit path. The other optimized the sum over all paths compatible with the target's physical codes. The latter likelihood can be evaluated by propagating prefix mass through deletion, insertion and diagonal edges, combining copy and replacement on a shared diagonal. This recurrence is exact for the declared code observations. Entire output images and coding collisions require separate physical evaluation. [Objective and proof](visual_alignment_enumeration.md) · [Training protocol](visual_span_alignment_protocol.md)

The common training data comprised 16,384 real classical–modern pairs from 3,552 chapters. We limited training lengths to 24 source and 32 modern glyphs to control quadratic lattice cost. Each arm received 61,440 prompted presentations and 1,347 updates in three epochs, processing 1,127,538 reference cells and 20,058,611 padded lattice cells. Both updated 568,373 of 1,984,053 parameters. On one RTX 4090 D the additional fits took 15.4 and 23.9 seconds, each peaking below 779 MiB of allocated GPU memory. Those costs exclude inherited training, preprocessing and evaluation. No matched-cost general-language benchmark has established a speed advantage over a text LLM. [Fixed fit](evidence/visual_span_20261118/fixed_fit.json) · [Marginal fit](evidence/visual_span_20261118/marginal_fit.json)

## Choosing and generating phrases

Two fresh rosters contain 1,024 requests each from 288 and 317 training-excluded chapters. Each classical source was deduplicated against previous frozen sentence rosters by string and physical sequence. Two and seven requests were unrenderable. The tests retain those failures in exact-answer denominators. The local comparison uses the same parent and new training pairs. All quoted renderable passages were copied exactly by the two fitted writers. [Roster receipt](evidence/visual_span_20261118/rosters.json)

| Writer / decoder | Exact discovery | Discovery edit rate | Exact replication | Replication edit rate |
|---|---:|---:|---:|---:|
| Local edit counts | 10 | 55.27% | 3 | 58.33% |
| Original visual editor | 8 | 55.72% | 7 | 58.85% |
| Fixed alignment, greedy actions | 5 | 176.67% | 4 | 161.27% |
| Summed alignments, greedy actions | 8 | 294.97% | 5 | 296.49% |
| Fixed alignment, most probable next image | 6 | 55.22% | 9 | 58.62% |
| Summed alignments, most probable next image | 6 | 54.96% | 9 | 58.43% |
| Summed alignments, estimated minimum edit risk | 6 | 55.13% | 9 | 58.41% |

The summed-path loss improved free recovery of complete recorded insertion spans from 365 to 484 out of 10,789 under greedy actions. Given the true cursor and preceding modern writing, the number of correctly drawn spans rose from 443 to 530. Three complete changed answers in discovery used the marginal writer's greedy actions. One transformed 官至岳阳王府谘议、东扬州别驾。 into the recorded 官到岳阳王府谘议、东扬州别驾。 On replication all five exact outputs were unchanged sources. Error rates above 100% reflect extra writing. In response to 而逢吉谋益露。 the writer emitted 192 glyphs and hit the output cap. Every one of its 191,987 freely drawn greedy translation glyphs across the two cohorts matched a bitmap in the scoring collection. [Saved-output comparison](evidence/visual_span_20261118/comparison.json)

Next-image decoding follows many source cursors at once. It sums paths through deletions and combines copy and draw mass for each successive measured ink pattern. Four surviving within-glyph prefixes approximate the best of the possible complete codes. For the marginal writer, the winning code or stop exceeded all pruned-prefix upper bounds on 46,552 of 47,117 translation decisions. All 2,039 renderable translations stopped. Its 15 exact answers were unchanged sources. The chapter intervals for improvement over local edits included zero on both cohorts. The initial implementation encountered a tensor-shape error before producing any outputs; after indexing the within-glyph bias by position, probability conservation held within 0.000001 and 40 stopped examples across both fits, both cohorts and both tasks agreed with the training likelihood within 0.00071 nats. [Decoding protocol](visual_span_decoding_protocol.md) · [Discovery](evidence/visual_span_20261118/marginal_prefix_discovery_report.json) · [Replication](evidence/visual_span_20261118/marginal_prefix_replication_report.json)

## What the order comparison establishes

Each phrase exchange has two different chapters, the same immediate modern neighbors, and matched phrase lengths. In 85 of 293 pairs the exchanged phrases contain multiple glyphs. For each pair the exact likelihood compares the sum of the two correct log probabilities with that of the two swapped targets. This controls for the unconditional frequency of the two offered phrases. The original and reversed source contain the same glyphs. The source-dependent change can still reflect local ordering or translation conventions. It does not establish a general representation of semantics.

| Marginal writer | Discovery (154 pairs) | Replication (139 pairs) |
|---|---:|---:|
| Recorded assignment, complete reference | 111 | 102 |
| Recorded assignment, reversed classical source | 81 | 80 |
| Recorded assignment, subsequent modern writing withheld | 112 | 98 |
| Subsequent writing withheld, reversed source | 79 | 69 |

With the complete references, source reversal reduced preference for the recorded assignment by 1.70 and 1.91 nats per pair. With subsequent writing withheld the reductions were 1.45 and 1.69 nats (95% pair-resampling intervals 0.85–2.12 and 1.14–2.30). The reader therefore had source-dependent information at the time the phrase was required. The four constructed pairs swapping agent and patient gave four correct joint assignments for each fit, with zero pairs of independently exact free answers. The eight sources were absent from the searched rosters. [Exchange protocol](visual_span_exchange_protocol.md) · [Prefix-only discovery](evidence/visual_span_20261118/discovery_prefix_exchange.json) · [Prefix-only replication](evidence/visual_span_20261118/replication_prefix_exchange.json)

## What sampling established

The two deterministic readers tended to repeat marks or return the input. We drew 32 trajectories per request to minimize expected edit distance directly. The first 16 proposed strings and a per-source-position consensus, alongside the existing deterministic answers. The other 16 estimated each candidate's risk without the reference. All 65,248 sampled translations reached a learned stop. They contained 1,950,760 emitted glyphs, of which 721,137 matched no bitmap in the single-font scoring bank. A new shape can be legitimate; exact absence from this bank measures fidelity to the pictured corpus. In 32 draws per request, any sample reproduced an entire changed reference on two discovery inputs and zero replication inputs. Minimum-risk selection returned no exact changed translations. All four corpus outputs and sample totals were recounted from saved pixels. A constructed consensus has a finite ending by construction. The sampled stops and the independently decoded stops provide the evidence about learned termination. [Risk protocol](visual_span_risk_protocol.md) · [Discovery recount](evidence/visual_span_20261118/discovery_sampling_audit.json) · [Replication recount](evidence/visual_span_20261118/replication_sampling_audit.json)

## Using and reproducing the writer

The public span parameter update runs with the original frozen renderer on CPU or a single GPU. Use the single-path and summed-prefix settings to reproduce the different outputs in Figure 51. The existing default editor remains preferable for the tested application. [Inference instructions and weights](../models/visual_edit_zh/README.md)

From the repository root with the original NiuTrans corpus, parent editor and previous frozen rosters available, use:

```sh
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python scripts/prepare_visual_spans.py
python scripts/train_visual_spans.py
python scripts/evaluate_visual_spans.py
python scripts/score_visual_prefix_decoder.py
python scripts/probe_visual_span_exchange.py
python scripts/probe_visual_span_exchange.py --through-phrase
python scripts/score_visual_span_risk.py
```

The resulting checkpoints, actual output archives, role tests and sampled trajectories live under `artifacts/visual_span_alignment_v1/`. The next learning comparison should give a proposer direct supervision on complete inserted phrases. It should read the original source and the available pictured prefix, draw the phrase and decide where it ends. Compare with the existing writer at the same known insertion sites before testing at inferred sites. Match real data and exposures, retain the local phrase-frequency control, test novel whole phrases, and score the complete surrounding sentence after integration. Choosing among offered phrases and producing a useful answer are separate acceptance criteria.
