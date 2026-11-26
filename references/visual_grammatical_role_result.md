# Teaching the visual writer where a verb is written

The independent 1,688,480-parameter writer learned to concentrate more attention on source glyphs carrying a matching grammatical role. On 384 translations of unseen book passages it earned ten source-only meaning credits. The likelihood-only control also earned ten. Both received seven credits with the supplied reference identified. The true-role writer passes the stopping and controlled-family criteria. The whole-meaning improvement criterion fails. [Registered comparison](library_role_grounding_protocol.md) · [Complete results and all votes by work](evidence/library_roles_v1/evaluation.json)

| Writer | Attention to matching source role | Permuted expectation | All eight exact variants, 118 | Source-only meanings, 384 | Reference-informed meanings, 384 |
|---|---:|---:|---:|---:|---:|
| Starting writer | 25.65% | 20.65% | 108 | 10 | 5 |
| Continued likelihood | 25.62% | 20.65% | 111 | 10 | 7 |
| Shuffled role labels | 29.27% | 23.94% | 110 | 9 | 6 |
| True role labels | 29.78% | 23.87% | 110 | 10 | 7 |
| Qwen3-8B Q4, text input | — | — | — | 315 | 198 |

The three fits used identical pictured passages, rehearsal and batch order. Each received 8,753,476 glyph or stop positions and 2,768 updates. The supervised fits each saw 3,094,812 role targets. A row-preserving permutation moved 442,383 of 878,416 source tags. Each additional four-pass fit took 34–40 seconds on a single RTX 4090 D from a pretrained visual checkpoint. All models use the same output bitmap inventory. [Training receipts](evidence/library_roles_v1/training.json) · [Released weights](../models/library_roles)

Source attention was measured with correct preceding target glyphs. On each of 8,249 eligible development positions the gradient of the auxiliary objective was independently checked. The maximum absolute discrepancy from automatic differentiation was 2.1 × 10⁻¹⁶. The maximum directional central-difference discrepancy was 1.8 × 10⁻¹⁰. Matching a role leaves the identities of words within that role unresolved. [Lean derivation](../formal/span_list/RoleLocationLearning.lean) · [Physical-network derivative check](evidence/library_roles_v1/gradient.json)

Each of four reserved works contributed 48 bilingual sentence pairs. Source-only credits for the role-trained writer were 2 from Gatsby, 7 from Plato, 1 from Water Margin and 0 from the bilingual Bible. All four visual writers stopped on every request. The source-only interval for the true-role improvement over likelihood was [0, 0] across work resamples. The interval against shuffled supervision was [0, 0.78] percentage points. The Qwen3-VL bilingual assessment used two candidate orders and source-only duplicate collapsing. Corpus references earned 219 and 383 credits in the two readings; unrelated controls earned zero in each. No independent human panel adjudicated the outputs.

All four writers translated the pictured “inquired Tom.” as “汤姆问道。”. On a longer Gatsby sentence the role-trained writer lost the apology and repeatedly wrote “straight”. The article includes both original source grids and unaltered output images. The complete archive contains 1,536 verified PNGs, the stop decisions, original input hashes, bibliographic coordinates and case-level grading outcomes. [Illustrated comparison](../publication/ilm-nature/figures/Figure_212_words_and_roles.png) · [All independent writer outputs](../publication/ilm-nature/evidence/role_translation_outputs.zip)
