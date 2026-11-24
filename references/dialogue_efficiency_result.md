# Can glyph images make language learning cheaper?

A writer trained entirely from random weights on glyph images learned to answer natural questions about a pictured passage. The most informative comparison changed only the letter observation. Two 547,125-parameter writers saw identical CoQA turns, used the same word boundaries and could copy and spell in exactly the same ways. One read measured 32×32 ink cells and the other read ASCII byte identities. The image model reached essentially the same performance and consumed slightly more energy at the common quality targets. [Protocol](dialogue_representation_protocol.md) · [Run either writer](../models/dialogue_surfaces/README.md)

| Model | Reserved primary exact / 4,413 | Development primary exact / 5,588 | Official development word F1 | Training time | Dialogue fit energy |
|---|---:|---:|---:|---:|---:|
| Visual writer with inherited reader | 1,256 | 1,747 | 46.7% | 132.6 s | 15.04 kJ |
| From-scratch subword Transformer | 519 | 708 | 17.3% | 126.5 s | 16.59 kJ |
| From-scratch visual writer | 1,103 | 1,574 | 41.8% | 133.2 s | 15.61 kJ |
| From-scratch byte writer | 1,121 | 1,558 | 42.2% | 131.8 s | 15.44 kJ |

Each fit used one RTX 4090 D, 8,000 batches of 32 examples and 71,410 distinct sampled training turns. Preparation energy, evaluation, host power and earlier supervised training are excluded from the last column. The subword Transformer had two encoder and two decoder layers, 817,506 parameters and a training-only BPE vocabulary. It generated every output subword. The visual and byte writers had a pointer as well as a spelling route. The difference between the first two rows therefore concerns two complete systems. The last two rows isolate how a letter is observed within this architecture. All models were trained at a single seed.

## Accuracy at the same training budget

| Updates | Ink, reserved | Byte, reserved | Ink, development | Byte, development |
|---:|---:|---:|---:|---:|
| 100 | 411 | 429 | 583 | 579 |
| 800 | 751 | 825 | 1,055 | 1,112 |
| 2,400 | 901 | 982 | 1,268 | 1,352 |
| 8,000 | 1,103 | 1,121 | 1,574 | 1,558 |

At the final checkpoint the development difference was +0.29 percentage points for images, with a 95% whole-story bootstrap interval from −0.77 to +1.35. The reserved difference was −0.41 points, with an interval from −1.58 to +0.77. Both arms first cleared 10% on both splits at update 800, using 2.21 and 1.87 kJ including feature preparation. At 20% both first cleared the target at update 2,400, using 5.20 and 4.86 kJ. The byte writer reached 25% on both splits at update 8,000. The visual writer fell one reserved answer short. These are checkpoint observations. The precise crossing between evaluations is unknown. This experiment provides no support for 10×, 100× or 1,000× energy savings. [Cold comparison and intervals](evidence/dialogue_ink_v1/cold_comparison.json) · [Subword comparison](evidence/dialogue_ink_v1/token_comparison.json)

## What the writer learned

Exchanging the current question and previous exchange on the same pictured story reduced the cold image writer from 1,574 to 237 exact development answers. Replacing the passage reduced it to 672. The byte writer scored 1,558, 187 and 686. On the 83 development turns whose four human answers all required an absent multiword phrase, all four systems scored zero exact. Answers produced on a held story demonstrate source-sensitive short-answer reading. Generating fresh explanations, handling arbitrary scans and learning more quickly than a strong pretrained language model remain open.

Lean proves that a losslessly rendered corpus preserves the attainable finite-sample prediction loss in both directions. Representation alone supplies no computational shortcut. A second proof shows that a deterministic writer generates a complete target image sequence exactly if and only if every word decision and the final stop are correct on the corresponding reference histories. An implementation checked this equivalence on 20,002 real visual replies with zero disagreements. Of the inherited direct writer's 4,367 development image mismatches, the first incorrect decision selected the wrong visible word in 3,404 cases, spelled the wrong word in 788, stopped early in 60 and continued beyond the reference in 115. An exact-image discrepancy can also be an acceptable alternative human answer. These first-error counts give a concrete training target for evidence selection. The separate Lean margin bound describes when a style perturbation preserves a decision. Handwriting robustness requires measuring its hypotheses on real inputs. [Lean theorems](../formal/span_list/VisualTokenRisk.lean) · [Physical verification](evidence/dialogue_ink_v1/plain/prefix_certificate.json) · [CPU replays](evidence/dialogue_ink_v1/comparison_cpu_replay.json)
