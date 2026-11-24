# A writer that can retrieve earlier word images

Asked where Mr. Preece worked, the visual writer returned **University of Bristol**. When given a different passage it wrote **in Marathon Athens**. A new memory could also change a one-word answer from **No** to the correct **yes**, although exchanging the question preserved that **yes**. On a question asking what deal two people made it wrote **he had with palms**, against the reference **would toss a coin**. These are standalone outputs replayed from PNG inputs. [Actual answer strips](evidence/word_memory_v1/examples.json) · [Run the released writer](../models/word_memory/README.md)

| Input and decoding | Exact on all 2,238 | After question/history exchange | After passage exchange | Exact on 259 difficult | Difficult word F1 |
|---|---:|---:|---:|---:|---:|
| Image, original parent | 500 | 55 | 262 | 0 | 14.40% |
| Image, merged scores, retrieval disabled | 502 | 58 | 243 | 0 | 14.07% |
| Image, retrieval enabled | 502 | 59 | 254 | 0 | 14.79% |
| Byte, original parent | 473 | 61 | 251 | 0 | 11.84% |
| Byte, merged scores, retrieval disabled | 479 | 68 | 238 | 0 | 11.47% |
| Byte, retrieval enabled | 473 | 68 | 254 | 0 | 11.49% |

The 147 previously unused CoQA stories were reserved before fitting, after excluding stories in earlier cohorts and the source registries. Each difficult reference requires a multiword phrase absent from the displayed inputs after normalization. With one fixed training seed the visual writer gained 0.39 percentage points of difficult word F1 relative to its parent (95% paired story-bootstrap interval −0.42 to 1.28). The difference against the decoder with retrieval disabled was +0.73 points (−0.26 to 1.95). Neither comparison yielded an additional exchange-sensitive difficult exact answer. Each question has one reference, which can miss valid alternatives. [Registered comparison](word_memory_protocol.md) · [Complete outcomes](evidence/word_memory_v1/results.json)

## Where the missing words went

With the reference length and every other answer word given, the visual writer correctly filled 356/1,100 source-copyable positions. It filled 0/108 when the correct word image was absent from the inputs but present in memory, and 0/100 when absent from both. The parent had exactly the same counts. Forcing retrieval would recover just 3/108 stored words. The correct image reached the top eight on 17 occasions and the top 32 on 38. Its median rank was 117 in the 2,048-word inventory and the median retrieval gate was 10.6%. Merely biasing the model towards retrieval cannot resolve this selection failure. [Word ranks and gate probabilities](evidence/word_memory_v1/stored_ranks.json)

The decoder merges probability mass over all source copies and memory entries drawing the same word, then adds the probability of one greedy spelling. A Lean proof shows when the leading word is certain even after adding all unexamined spellings, and proves that the margin is tight given only the remaining mass budget. The numerical check passed on 678/1,308 held visual slots. This is a claim about maximizing the model's assigned word probability. The model can confidently favor an incorrect answer. [Checked proof](../formal/span_list/WordMemoryMode.lean) · [Decoder and objective](../ilm/visual_lm/word_memory.py)

## Training and cost

The 785,190-parameter writer includes its inherited body and 211,169 newly fitted parameters. The complete visual checkpoint is 4,264,184 bytes. It contains 2,048 output word prototypes selected from fitting answers, a finite image vocabulary along one output route. The writer can also copy input word images and spell observed glyphs. All experiments use aligned ASCII English print with explicit word gaps.

Both new fits used the same 102,400 presentations (44,417 distinct training turns) on GPU 0, an RTX 4090 D. Preparing and fitting the visual model used 0.49 and 2.86 kJ, respectively. Its optimizer ran for 24.4 seconds with 347 MB peak allocated GPU memory. The byte fit used 2.88 kJ in 24.6 seconds. Across the four actual ancestral checkpoints, the sum of measured preparation and optimizer device energy is 31.12 kJ for images and 29.96 kJ for bytes. These totals omit workstation power and offline corpus construction. The trial establishes neither a matched-quality efficiency advantage nor general open-ended reading. [Checkpoint lineage and held-evaluation costs](evidence/word_memory_v1/cost.json)

The next comparison should determine whether contextual training specifically on a concealed, source-absent word improves its rank against frequent but wrong fitting words. Test against a frequency-only prior, the frozen reader, and question and passage exchanges. Require better conditional discrimination on unseen stories before investing in a larger word memory. Preserve a separate autonomous composition test.
