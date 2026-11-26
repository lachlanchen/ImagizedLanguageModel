# What the writer learns from glyph order

Exchanging two people in a pictured passive sentence can change the ILM's written Chinese answer correctly. On natural bilingual sentences the same family of models uses glyph order to recognize some correct translations, yet loses meaning when it has to compose the reply. The article shows complete input and output grids, including the first held scientific sentence in both directions. [Original images and replay hashes](evidence/paired_ink_v1/raw_rasters.json) · [Reading edition, pp. 45–46](../publication/ilm-nature/visual_language_paper.pdf#page=45)

## Matched training and free generation

Two optical writers started from the same 1,688,480-parameter varied-source checkpoint on one RTX 4090 D. Both trained on 2,879 non-equivalent PAWS-X sentence families and ordinary parallel replay, in both directions, with identical batches and 92,128 positive presentations each. Chinese PAWS-X training sentences were machine-translated. One fit added a loss for assigning two supplied translations to the right sources. We fixed the final checkpoints and reserved 48 other official test families before training. The two directions and two variants per family give 192 free requests. Complete-sentence overlap was screened after normalization in both languages. This test uses pictured text in a fixed font and cell layout. [Protocol](paired_ink_protocol.md) · [Split and hashes](evidence/paired_ink_v1/split.json) · [PAWS-X](https://aclanthology.org/D19-1382/)

| System | Correct assignments / 96 | Both sources independently prefer their translation / 96 | Mean assignment loss (bits) | Complete meanings / 192 |
|---|---:|---:|---:|---:|
| Starting writer | 77 | 31 | 0.926 | 0 |
| Translation likelihood | 75 | 26 | 1.651 | 0 |
| Likelihood and assignment | 75 | 26 | 1.648 | 0 |
| Quantized Qwen3-8B, text input | — | — | — | 135 |

Every ILM answer stopped. The local Qwen3-VL grader saw candidate identities masked and judged two presentation orders. It accepted 189/192 reference translations and 18/192 contrasting foils twice. These are model judgments. The paired 95% bootstrap interval over families for the added loss spans −3.1 to +3.1 percentage points in assignment accuracy. Held target loss rose from 2.146 to about 2.520 nats per target position, including stop, while training assignment loss fell from 0.0922 to 0.0026 nats in the intervention. [Scores for all assignment decisions, paired intervals and grading calibration](evidence/paired_ink_v1/scores.json)

## Information retained by the visual sequence

For source images X and candidate complete answer images Y, let sᵢⱼ be log p(Yⱼ | Xᵢ), summed through stop. The log odds of the matched assignment under equal prior odds and independent conditional draws is D = s₀₀ + s₁₁ − s₀₁ − s₁₀. This is a two-state Gibbs distribution when energy is defined as negative log probability. Its binary cross-entropy is log₂(1 + exp(−D)) bits. The measured loss applies to this candidate set. It does not estimate the entropy of English or Chinese.

Lean verifies that source-only offsets and target-only fluency cancel. It also verifies that two sources mapped to one representation force a tie, that a deterministic decoder then writes the same reply, and that an arbitrarily large margin on two references can coexist with a third incorrect answer scoring higher. The score algebra uses fixed-point integer units. The probabilistic model and its assumptions are stated separately. [Lean statements and proofs](../formal/span_list/PairedInkOdds.lean)

Sorting the complete source bitmaps by their bytes reduced correct decisions from 75 to 51 of 96. Eleven paired sources became identical after sorting. The model chose correctly for all eleven in reading order. After sorting, every one tied and produced identical free replies. On the other families, sorting can disrupt several cues simultaneously. [Order intervention and individual scores](evidence/paired_ink_v1/order.json)

Amplifying the next-glyph score attributable to intact reading order changed 192/192 PAWS-X outputs and 92/96 OPUS outputs. Neither the ordinary nor the modified decoder received a complete-meaning credit on either panel. In this separate grading run Qwen received 129/192 and 49/96 credits, respectively. The OPUS examples had appeared previously in retrieval evaluations. Retain the starting writer for its controlled successes. The principal remaining failure is generating the proposition in an unfamiliar sentence. [Inference rule and data reservation](ink_order_decoding_protocol.md) · [Free-writing results](evidence/paired_ink_v1/order_decoding.json)
