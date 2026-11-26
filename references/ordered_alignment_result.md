# Following source order in visual translation

Can the ability to locate a translation in a book help a model write it? We fitted a source-position alignment model to **497,775 parallel sentence images**. Among 48 candidate passages, the ordered model placed the correct source first for **91/96** new queries. The unordered control also scored 91. When each model had to write the translation itself, neither conveyed a complete meaning on the same 96 requests under the registered assessment. [Manuscript](../publication/ilm-nature/manuscript.pdf) · [Original source and answer images](../publication/ilm-nature/evidence/ordered_alignment.zip) · [Run the writer](../models/ordered_alignment/README.md)

## What was learned

A position in the source corresponds to a contiguous strip of glyph pixels. Across a translation, that position can be sampled uniformly, kept, or moved by −1, +1 or +2 strips, clipping to the source boundaries. Let $A_{ij}$ be the transition probability. The state starts uniformly. For a source of length $n$, the fixed emission law at position $j$ is $e_j(z)=[nT(z\mid s_j)+T(z\mid\varnothing)]/(n+1)$, where $T$ is the previously fitted bilingual glyph-strip table. If every transition is uniform, averaging $e_j$ gives the earlier unordered model. The transition law was estimated by expectation maximization on paired visual strips. This is a glyph-stream application of hidden Markov alignment. [Earlier lexical model](full_image_roles_result.md) · [Vogel, Ney and Tillmann](https://aclanthology.org/C96-2141/)

With $f_{k-1}$ the filtered position probabilities, a target strip $z_k$ gives $u_j=e_j(z_k)\sum_i f_{k-1,i}A_{ij}$. The sum of $u$ is the conditional probability of the strip. Normalizing it gives the updated position distribution. A backward pass accumulates expected uses of the five movement types. For free image generation the reader maintains these distributions at every possible strip boundary in the emitted glyph prefix. The existing neural logit correction and the probability of stopping are preserved. At inference there is no supplied translation, token transcript or language-model service. The writer chooses glyphs from a fixed printed inventory, and the image interface uses aligned 32×32 cells. [Code](../ilm/visual_lm/ordered_alignment.py) · [Pre-fit protocol](ordered_alignment_protocol.md)

| Target | Uniform reset | Back one | Stay | Forward one | Forward two | Target strip positions per pass |
|---|---:|---:|---:|---:|---:|---:|
| English | .239 | .042 | .380 | .257 | .082 | 7,502,810 |
| Chinese | .349 | .035 | .158 | .295 | .164 | 5,610,631 |

Each direction completed four full passes. The GPU fit took 23.1 s for English and 18.5 s for Chinese and peaked below 0.46 GiB of allocated CUDA tensors. These timings begin after corpus preparation and device loading, and exclude the previous training of the bilingual table and the 1.92M-parameter visual reader/writer. The complete-system matched-quality compute advantage remains open. [English fit](../models/ordered_alignment/en/fit.json) · [Chinese fit](../models/ordered_alignment/zh/fit.json) · [Corpus binding](evidence/ordered_alignment_v1/corpus.json)

For two source positions, the difference in joint weight of two target orders factors into the difference between the forward and backward transition masses times the difference between the two cross-position emission products. With a uniform starting distribution, the common initial factor cancels in the comparison. Lean verifies this identity and the sufficient condition for strict preference at shared integer scales. The numerical implementation's likelihood and expected counts agreed with full path enumeration to $6.7\times10^{-16}$ in the GPU check. On 254 glyph prefixes, the largest probability discrepancy versus independent enumeration was $3.4\times10^{-16}$. Twelve original answers replayed byte-for-byte at the pixel level on CPU, including their stop decisions. [Lean theorem](../formal/span_list/OrderedStripLikelihood.lean) · [GPU check](evidence/ordered_alignment_v1/kernel-check.json) · [Glyph and replay audit](evidence/ordered_alignment_v1/audit.json)

## Free writing and search

The four new works were *Hopscotch*, *Ancient Egypt History*, *Japanese History* and *Samurai World Warrior*, identified by edition hashes in the corpus record. The first twelve eligible aligned passages from each yielded 48 distinct pairs and 96 directional requests. The reservation excluded exact training-side bitmap matches. The 48-way search has one strict first-place opportunity per query and discards strips containing only script-neutral marks. Earlier and fresh panels use different works. [Reservation](evidence/ordered_alignment_v1/corpus.json) · [Search records](evidence/ordered_alignment_v1/retrieval.json)

| Held-out measurement | Neural writer | Unordered alignment | Ordered alignment |
|---|---:|---:|---:|
| Complete free translations, new books / 96 | 0 | 0 | 0 |
| Natural stops, new books / 96 | 96 | 96 | 96 |
| Both polarities exactly right / 236 object-cleft pairs | 195 | 207 | 205 |
| Both roles and both polarities exactly right / 118 sets | 84 | 92 | 91 |
| Correct source first, prior 48-way panel / 96 | — | 93 | 94 |
| Correct source first, new 48-way panel / 96 | — | 91 | 91 |
| Distinct reference strips reproduced / 1,416 | 207 | 237 | 236 |

The same source-only evaluator scored the text-input quantized Qwen3-8B at 81/96, the provided bilingual references at 63/96, and unrelated foils at zero. Repeated identical candidate texts were assessed together. Credit required agreement across opposite presentation orders and a natural stop. The evaluator is an automated bilingual model. It rejected 33 supplied translations, and fragmentary sources and disagreements over wording can affect its judgment. The 96 requests share four works and include both directions of each pair. Literal strip overlap can penalize paraphrases. [Complete assessments, by work and language](evidence/ordered_alignment_v1/assessment.json)

The word order intervention changed the ordered writer's response to *“Cold as hell,” said Gregorovius.* It recovered 冷 and replaced the speaker with 绿色 (“green”). On a controlled prompt, *It was the student whom David did find*, its added 了 made the answer exactly match the reference. The unordered answer already described the finding event. The figure shows both comparisons directly from the saved PNGs. [Editable figure](../publication/ilm-nature/figures/Ordered_glyph_alignment.pptx) · [Figure provenance](evidence/ordered_alignment_v1/plate.json)

## Where writing lost meaning

After observing repeated phrases, we counted exact repeated triples of script-bearing strips in the 96 answers. The neural writer repeated a triple in 16, the unordered writer in 25, the ordered writer in 24, and the paired references in one. Truncating all four streams to the shortest strip count within each request gave 14, 22, 18 and one, respectively. This exploratory measure records literal repetition; repeated motifs can be legitimate. The ordered writer covered only 144/833 English reference strips and 92/583 Chinese strips. Removing repeats alone would leave the missing source content unresolved. [Counts for every answer image](evidence/ordered_alignment_v1/repetition.json)

The ordered model failed the registered free-writing criterion. Preserve both search models. The next generative comparison should learn jointly observed bilingual **phrases**, consuming a recorded source span when emitting its target span so that a writer can track what it has translated. Compare free generation, coverage of names and substantive content, and legitimate as well as erroneous repetitions on newly reserved books. Keep the original ordered and unordered models as controls. Demonstrating a reduction in repeated strings alone would leave the language objective open.
