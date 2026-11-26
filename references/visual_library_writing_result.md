# What transfers to a new book?

A 1,688,480-parameter optical writer trained on 28,342 bilingual literary pairs from 21 works drew the exact English answer **“But who is he?”** from a pictured Chinese question. Across 960 independently generated translations from four reserved novels it reproduced six references and stopped on every request. It still repeated words and lost relationships on longer prose. The first Kafka sentence asks the reader to compare one of the father's steps with the many movements needed by Gregor. The model repeats “father” and loses the comparison. [Raw questions and outputs](../publication/ilm-nature/figures/library_diversity) · [Runnable weights](../models/library_diversity)

## Training diversity and free translation

A cycle sampler reduced the training set to 4,896 distinct pairs while preserving both language length histograms in each work. The broad and repeated arms saw the same number of target positions in each minibatch and the same replay. Four passes gave each 2,768 updates, 8,753,476 supervised positions and 265,648 stops, taking 43.7 and 42.4 seconds on one RTX 4090 D. These are additional fits from an existing checkpoint. The sampling entropy fell from 10.2521 to 8.3651 nats in the repetition control. Lean verifies the mean-preserving update and its linear-score and margin identities. [Sampling and fit protocol](visual_library_diversity_protocol.md) · [Sampler receipt](evidence/library_diversity_v1/sampling.json) · [Lean proof](../formal/span_list/PairedLibrarySampling.lean)

| Writer | Exact image and stop, 960 | Mean stopped glyph similarity | All eight role variants, 118 | Whole meanings on first 96, source-only / reference-informed |
| --- | ---: | ---: | ---: | ---: |
| Starting | 2 | .309 | 111 | 1 / 1 |
| Repeated passages | 2 | .286 | 69 | 1 / 1 |
| Distinct passages | 6 | .327 | 108 | 1 / 1 |
| Text Qwen3-8B Q4 | — | — | — | 82 / 55 |

The automated bilingual judge saw the source and a shuffled set of candidate answers in each of two orders. The source-only reading deduplicated identical candidate text. The reference-informed procedure identified the supplied corpus translation. Neither involved human adjudication. The four held works were *The Metamorphosis*, *Germinal*, *The Odyssey* and *Within a Budding Grove*. The natural-language improvement criterion failed. [All scores and per-work intervals](evidence/library_diversity_v1/evaluation.json)

## Source evidence and associative memory

The source-contrast intervention divides the model's glyph preference by a bounded background term measured on four training sources at the same answer history. Calibration on both development books chose zero contrast. The positive risk derivative at zero (+0.06565) agrees with the observed convex increase in negative log probability. A fixed diagnostic gain of 0.5 changed 772/960 answers. It turned “Then you have a money?” into “Then you have money?” but lost one stop and retained 108/118 role families. Its paired interval includes zero. The fixed-point score bound is proved in Lean. [Protocol and derivation](library_ink_contrast_protocol.md) · [Raw scores](evidence/library_diversity_v1/contrast.json) · [Lean certificate](../formal/span_list/BoundedGlyphEvidence.lean)

The associative writer stores the next glyph after 1,835,814 real training states. It retrieves 32 source passages and 16 writing states, assigning a Gibbs weight to each next bitmap. The two directional archives require 613,398,604 bytes. Construction and calibration took 20.1 seconds on GPU0. At the selected temperature of 0.05 and mixture weight 0.10, validation loss fell from 1.76148 to 1.75957 nats per glyph or stop. Shuffling the labels within preceding-glyph groups raised that loss to 1.80352 at the same settings. The source-only meaning counts in the table use three disjoint sets of 96 requests. The comparisons share the four test works. [Memory protocol](library_associative_memory_protocol.md) · [Calibration](evidence/library_memory_v1/calibration.json)

| Comparison | Ordinary or matched control | Intervention | Qwen | Intervention with identified reference |
| --- | ---: | ---: | ---: | ---: |
| Distinct passages | 1 | 1 | 82 | 1 |
| Source contrast at 0.5 | 3 | 4 | 79 | 3 |
| Associative memory | 3 | 3 | 84 | 0 |

All three memory conditions earned three source-only credits and zero reference-informed credits. All 960 memory outputs stopped. The memory writer retained 109/118 complete role families. It recovered “beer” in a sentence where the ordinary writer produced “wine”, but repeated “beer” inside an unfaithful reply. None of the three interventions qualified for promotion. No matched-quality acceleration can be inferred. [Memory outputs and grades](evidence/library_memory_v1/evaluation.json)

## Can retrieval supply the phrase?

We measured exact continuation availability from the correct supplied history. An English candidate begins at a word start and contains three or more letters. The Chinese diagnostic uses four consecutive Han glyphs and allows overlaps. The denominators are not interchangeable.

| Source of continuation | English words out of 1,719 | Four-Han runs out of 2,348 |
| --- | ---: | ---: |
| Actual 16 retrieved states | 515 | 27 |
| Any position in 32 retrieved passages | 924 | 32 |
| Any position in 32 random passages | 687 | 4 |
| Anywhere in the 21 training works | 1,613 | 368 |
| One or two library fragments of at least two glyphs each | 1,697 | 1,560 |

Availability uses the reference to identify a correct span. The remaining challenge is selecting and combining fragments from their meaning in the source. As a source of supervision, the trilingual books attach grammatical roles to each writing unit. Exact corpus joins found 569,414 English-target and 204,289 Chinese-target glyph positions for which a subject, predicate or object is also present on the opposite side. These inherited annotations have not been independently adjudicated. [Phrase support](evidence/library_memory_v1/phrase-support.json) · [Global coverage](evidence/library_memory_v1/library-coverage.json) · [Exact role alignment](evidence/library_roles_v1/alignment.json)
