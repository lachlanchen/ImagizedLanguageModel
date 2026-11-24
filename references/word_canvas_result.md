# What the visual writer learns from a blank answer strip

A 574,021-parameter masked visual writer filled a two-word answer with **Amanda** on its first pass and **Lenhart** on its third. Exchanging the pictured question and history changed its answer to **15-year-old**. Exchanging the passage changed it to **Christopher Reeve**. The held questions also expose its mistakes. Asked “Which led to what?”, it wrote **his and films** for **breaking his back**. The left-to-right visual control answered a different difficult question with **he killed himself**, assembling three visible words in an order absent from the inputs. No complete fitting answer had that exact wording after lowercasing and whitespace cleanup. [Example pixels and source identifiers](evidence/word_canvas_v1/examples.json) · [Checkpoint and instructions](../models/word_canvas/README.md)

| Observation and writer | Exact on all 3,155 | Exact after question/history exchange | Exact after passage exchange | Exact on 443 difficult | Difficult word F1 |
|---|---:|---:|---:|---:|---:|
| Ink, parent | 769 | 76 | 332 | 0 | 15.12% |
| Ink, causal continuation | 743 | 78 | 337 | 1 | 15.54% |
| Ink, masked continuation | 693 | 63 | 353 | 0 | 13.03% |
| Byte, parent | 772 | 82 | 342 | 0 | 13.53% |
| Byte, causal continuation | 734 | 79 | 345 | 0 | 14.99% |
| Byte, masked continuation | 657 | 76 | 345 | 1 | 13.87% |

The 200 held CoQA stories came next in the earlier reserved hash ordering. Every input-eligible question entered the denominator. The single reference can miss valid paraphrases. The difficult visual F1 difference between masked and causal continuations was −2.51 percentage points (95% paired story-bootstrap interval −4.52 to −0.44, one training seed). Both difficult exact successes in the table lost credit under each exchange. [Reservation](evidence/word_canvas_v1/reservation.json) · [Complete outcomes](evidence/word_canvas_v1/results.json)

## Proposal coverage and the first wrong word

Starting from its own eight predicted lengths and four fixed fill orders put four of 279 copy-assemblable difficult references among the first 32 visual proposals. Two ranked in the first eight and none ranked first. The frozen parent's copy search found zero in 32 on these same new turns. Each exchange reduced the four visual proposal matches to zero. Across 164 additional difficult turns requiring new normalized words, no masked proposal matched. The search budgets were equal, but their action spaces and exploration rules differ. [Saved proposals and replies](evidence/word_canvas_v1/responses)

The visual masked writer predicted the correct word count on 91/443 difficult turns. Supplying the reference length produced only one normalized match and zero literal matches among 435 references within the output geometry. All 435 eventually acquired an incorrect literal word. The first such commitment occurred on pass one in 393 cases, pass two in 23, pass three in 14 and pass four in five. The Lean proof gives a certificate for this interpretation. For a target to remain possible under immutable commitments, every committed word must agree with its reference and every remaining slot must individually admit the target. The conditions are sufficient for independently fillable slots. A neural writer can have further dependencies. [Diagnosis](evidence/word_canvas_v1/diagnosis.json) · [Lean proof](../formal/span_list/IrrevocableDraft.lean)

With all other human answer words visible, the masked visual writer recovered 677/1,937 exactly copyable words and 0/327 source-absent word images. On these 327 it selected the spelling route ten times. Forcing its speller would produce only nine correct word images. Giving the reader the right answer length or changing the route choice alone has limited scope here. The fitting answers contained 245 of those 327 word occurrences. A bank holding the most frequent 2,048 physical words would cover 152 occurrences and all missing words in 84/228 affected answers. These counts describe the available ink; learning when to use it still requires a held test. Changing pixels concealed by the mask left all predicted logits unchanged on a checked CPU/GPU input. Changing a visible word on the right changed the pointer scores on its left. [Physical word inventory](evidence/word_canvas_v1/word_support.json) · [Inference checks](evidence/word_canvas_v1/inference_checks.json)

## Cost and decision

Each continuation used 102,400 presentations (44,437 distinct turns) on one RTX 4090 D. The masked visual fit used 54.7 seconds, 6.95 kJ and 478 MB peak PyTorch allocation. Its causal control used 50.4 seconds and 6.65 kJ. The masked and causal byte fits used 52.8 seconds/6.87 kJ and 47.4 seconds/6.47 kJ. These optimizer costs exclude the parent, fitting-data preparation and evaluation; per-stage costs are released with the checkpoints. The registered masked training comparison failed its transfer criterion. The cost and quality measurements support no computational advantage for the masked visual writer. [Protocol](word_canvas_protocol.md) · [Full fit records](../models/word_canvas)

The next comparison should let the writer select an image from earlier fitting answers when the source omits a word, while retaining the ability to spell unfamiliar ink. Test both correctly chosen words and complete answers on new stories. The low recovery of copyable words under supplied neighbors remains a second constraint on whole-answer quality. Historical Chinese generation and general open-ended writing require their own measurements.
