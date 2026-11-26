# Which parts of a pictured sentence reach the answer?

Two visual systems read English and Chinese on one RTX 4090 D. A writer that retains the entire sentence can express polarity in Chinese. A learned bilingual lexicon can find the source image of a translated book passage among 48 candidates. Both systems still struggle to compose an unrestricted translation. Figure 1 shows a successful affirmative–negative pair and a lexical correction that changes an affirmation to “however”. Every source and response in that figure is an original PNG. [Paper](../publication/ilm-nature/manuscript.pdf) · [Editable figure](../publication/ilm-nature/figures/Full_image_role_writing.pptx)

## Writing with the source intact

The preceding three-head interface erased negation in 194/236 contrasts. Here the writer receives every source glyph. A separate 227,843-parameter marker estimates the probability that each cell belongs to a predicate, actor or undergoer. It takes the pictured sentence without a marked verb. Subjects and objects of several events can occupy the same window. The masks describe their union. English training uses annotated antecedents where available. Chinese supervision uses eligible transitive heads from UD Chinese GSD, normalized to simplified Chinese when offsets can be preserved. The marker saw 8,480 English windows and 1,682 Chinese windows during training. [Corpus and data revisions](evidence/full_image_roles_v1/corpus.json) · [English source attribution](evidence/natural_roles_attribution.md) · [Chinese GSD](https://universaldependencies.org/treebanks/zh_gsd/)

| Held-out marker | Predicate precision / recall | Actor precision / recall | Undergoer precision / recall |
|---|---:|---:|---:|
| English, 1,124 windows | 67% / 93% | 46% / 84% | 48% / 88% |
| Chinese, 208 windows | 45% / 68% | 39% / 61% | 45% / 72% |

Both 1,688,960-parameter writers start from the same earlier book model. Each makes 5,052 updates over two passes through 237,577 pairs, with 36,780,166 target glyph or stop positions. The corpus includes genuine bilingual book and talk material, earlier equal-length event replacements, and 6,240 controlled training pairs. The two fits took 104.1 and 104.4 GPU seconds and peaked at 1.07 GiB of allocated Torch memory each. The marker fit took a further 10.6 seconds. These are continuation costs. The base models, original corpora, preparation, evaluation and earlier training are additional costs. At inference the marked system uses 1,916,803 learned neural parameters and a fixed picture inventory. [Matched schedule](evidence/full_image_roles_v1/audit.json) · [Marker fit](../models/full_image_roles/marker/fit.json) · [Writer fit](../models/full_image_roles/marked/fit.json)

| Source construction | Full source alone | Full source with inferred fields | Same writer with displaced fields |
|---|---:|---:|---:|
| Ordinary clause, exact / 472 | 471 | 471 | 472 |
| Subject cleft, exact / 472 | 325 | 327 | 321 |
| Object cleft, exact / 472 | 428 | 430 | 428 |
| Both polarities on each object cleft / 236 | 193 | 195 | 192 |
| Both role orders and both polarities on object clefts / 118 | 83 | 84 | 81 |
| Complete novel book translations / 96 | 1 | 1 | 1 |

The same affirmative and negative event now have distinct visual inputs in every tested contrast. The checked Lean result proves preservation of source identity and equality of the attainable losses of unrestricted deterministic readers before and after adding a deterministic mark. The three fields introduce an inductive bias. Whether a finite writer benefits from that bias is empirical. Moving the fields changed 24/96 book answer images, 28/3,953 winning decisions under correct target history, and zero complete-meaning credits. The added field's summed vector norm was 0.63% of the pre-attention source-state norm on the book panel. [Lean proof](../formal/span_list/IntactPicturedSource.lean) · [Field sensitivity](evidence/full_image_roles_v1/influence.json)

For the pictured source 他想：我发疯了。, the marked writer emitted *He thought: I was crazy.* For 我上楼回到卧室。, it emitted *I went back to the bedroom.* The latter loses the upward movement and failed the strict automated criterion. The first panel covered twelve dialogue or prose units from each of four previously reserved-out works, in both directions. The quantized Qwen3-8B text comparison earned 84/96, supplied references 68/96 and unrelated foils 0/96. The marked translation retained none of the eight previously tested GENTLE events in a single unblinded bilingual reading. The former prompted event extractor and the new complete-sentence translation interface have different tasks. [All 4,560 answer images](../publication/ilm-nature/evidence/full_image_roles.zip) · [Book votes and controlled counts](evidence/full_image_roles_v1/assessment.json) · [GENTLE readings](evidence/full_image_roles_v1/external-assessment.json) · [GENTLE licence and source](https://github.com/UniversalDependencies/UD_English-GENTLE/tree/12cbbbbf9edef2036309cb3bb8e8db346cdad815)

## Learning which English and Chinese marks correspond

The second experiment uses 497,775 distinct parallel image pairs from OPUS, books and talks. The two inventories contain 2,127 English and 5,749 Chinese entries including the reserved index. Each visible strip contains between one and seven cells. The fit uses physical glyph addresses and longest-match segmentation. For source strips $s_1,\ldots,s_m$, let $s_0$ denote a null alignment. For each direction we fit a conditional table $T$ by five iterations of expectation maximization. The probability of a target strip $z$ is

$$q(z\mid s)=\frac{1}{m+1}\sum_{i=0}^{m}T(z\mid s_i).$$

The expected training count for alignment $i$ is proportional to $T(z\mid s_i)$. The row update includes two unigram pseudocounts in total. Each released row stores its 128 largest probabilities, with the remaining mass distributed in proportion to the target prior over the remaining strips. The two compact tables occupy about 5.1 MiB on disk. Their numerical fits took 3.75 and 3.32 seconds, excluding segmentation, data preparation and the earlier inventory training. The likelihood follows IBM Model 1. A Lean proof establishes why permuting the already segmented source strips leaves its alignment mixture unchanged. This identifies a structural limit for role order even when content correspondence is strong. [Brown and colleagues](https://aclanthology.org/J93-2003/) · [Fit protocol](optical_lexicon_protocol.md) · [Lean theorem](../formal/span_list/OpticalLexicalOrder.lean) · [Model and deployment](../models/optical_lexicon/README.md)

The next 12 eligible passage pairs per work were reserved before the fit, with both pictured sides excluded from its training set. In 96 bidirectional comparisons the observed reference assigned more probability to its actual source than to a source from another work. When each target image was used to search for its source among **all 48 pictured passages**, the correct image was uniquely first on **93/96** requests. The result was also 93/96 within the same work among 12 candidates. A direct multiset overlap of physical glyphs scored 6/96 in both tests. The lexical score excluded strips without a letter in English or a Chinese character in Chinese. Fixed repertoire metadata supplies those script flags. A second implementation reproduced the 48-way result using just the archived input PNGs. [All scores and tied controls](evidence/optical_lexicon_v1/retrieval.json) · [Raw-PNG search replay](evidence/optical_lexicon_v1/pixel-search.json)

## Composing a new translation

For writing, the lexical model scores a glyph prefix by summing probability across every compatible boundary between complete and incomplete strips. At each step we add a clipped conditional log-probability ratio between the source-conditioned strip model and its unigram counterpart to the neural glyph scores. We normalize the ink choices while preserving the neural stop probability. A weight of 0.5 minimized development loss across 72 pairs. On 255 enumerable prefixes the implementation agreed with direct enumeration to a maximum absolute discrepancy of $3.34\times10^{-16}$. All 12 sampled independent CPU replays reproduced the GPU answer pixels. [Calibration](evidence/optical_lexicon_v1/calibration.json) · [Numerical and image audit](evidence/optical_lexicon_v1/audit.json)

| Second book panel, 96 requests | Complete meanings |
|---|---:|
| Frozen marked neural writer | 1 |
| Same writer with its matched lexical model | 1 |
| Same writer with another passage's lexical model | 2 |
| Qwen3-8B Q4, text input | 81 |
| Provided bilingual reference | 68 |
| Unrelated passage | 0 |

The two visual credits occurred on different requests. On *Yep, the United States*, the original writer emitted the reference exactly. The corrected writer kept 美国 but replaced 没错 with 然而. On *Science is nice*, the outputs were 科学是非常好的。 and 科学是很好的。 The automated judge accepted the latter alone. Its decisions are sensitive to intensification and translation choices, and it rejected 28 supplied references on this panel. Both conditions were assessed in both shuffled presentation orders and had to stop autonomously. The lexical comparison supplies evidence for cross-language image search. It establishes no general free-writing or matched-quality compute advantage. [Full blind assessments](evidence/optical_lexicon_v1/assessment.json) · [288 original answers and their inputs](../publication/ilm-nature/evidence/optical_lexicon.zip)

The next experiment should require a new proposition to be assembled in the correct order. Extend the alignment state to track the previous source position and measure whether the writer translates the same content words under changed order or negation. Keep the bag-of-strips mixture and the full-source neural writer as controls and reserve new works before fitting. Use complete meanings, coverage of distinct content words, natural stopping and raw answers as separate outcomes.
