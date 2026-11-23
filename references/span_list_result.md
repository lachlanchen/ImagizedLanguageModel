# What a pictured answer selector can recover

A scorer trained on 16,181 screened questions returned **234 exact image answers on 359 development questions**. The generic reader returned 242 and a matched model restricted to the first proposal from each reader returned 238. Keeping eight proposals per reader made 56 additional reference strips available. The trained model returned 12 of them. Against the generic reader it gained 16 exact answers and lost 24 (two-sided paired test, p = 0.268). All five performance comparisons for the replacement fit missed their prescribed thresholds. The wider reader is available as an optional checkpoint. [Original scores](evidence/span_list_screened_20261124/results.json) · [Run the model](../models/span_list_screened/README.md)

| Development questions | Generic | Previous selector | Matched depth one | Depth eight | Reference in wider list |
| --- | ---: | ---: | ---: | ---: | ---: |
| CMRC, 192 paragraphs | 112 | 112 | 112 | 111 | 161 |
| DRCD, 167 paragraphs | 130 | 130 | 126 | 123 | 151 |
| Total | 242 | 242 | 238 | 234 | 312 |

The first recovered lower-ranked answer supplied both requested game modes, **「战史演武」&「争霸演武」**, where the generic reader supplied only **战史演武**. The first lost exact match was informative: the answer **大律师余若薇** names the correct person, while the human reference gives **余若薇** without her professional title. Of 125 exact-match errors, 78 had an accepted answer elsewhere in the candidate set. [Original question and reply images](evidence/span_list_screened_20261124/examples.json)

### What reaches the writer

The independent factual writer emitted exact, stopped replies in both requested word orders on 15/29 eligible human factual questions and 925/1,200 DuIE passages. It passed all required replies on 70/128 name exchanges and 36/48 reordered boards. The likelihood-only specialist had passed 79 and 37. On 32 historical name substitutions the wider reader with the induced statement template reproduced 13 references, compared with 17 using generic object selection. In freer writing, 58/64 responses stopped, 45 contained both names, and none exactly reproduced the separate authored source clause. Other faithful phrasings are possible.

We changed the year in 32 actual source passages and retained the named fact. Given the correct same subject, relation and object, the deterministic writer emitted identical pixels for every pair and returned neither version's requested year correctly. The direct reader given each page and a pictured year question answered 5/64 versions and both versions on 2/32 pairs. Lean verifies why identical writer inputs cannot satisfy mutually exclusive answer obligations. Passing extra source evidence to the generator is necessary to distinguish these pages. [Date comparison](evidence/span_list_screened_20261124/details.json) · [Lean proof](../formal/span_list/WriterContext.lean)

### Source correction

Earlier evaluations used the official DRCD test split, which the project had reserved. At least 530 distinct reserved questions on 529 paragraphs were evaluated across the named records. Earlier scores on these inputs cannot establish untouched confirmation. The first wider-list trial also included seven training rows sharing a long passage segment with reserved material. We archived that trial, screened identities, article titles and text overlap, removed 547 training and 48 validation rows, and refitted all three heads. The replacement test uses eligible official development paragraphs. The original whole-pool independence claim has been withdrawn. Earlier backbone exposure remains unverified. [Exposure audit](drcd_source_exposure_correction.md) · [Replacement splits](evidence/span_list_screened_20261124/corpus.json)

### Training and reproduction

Each of four endpoint heads proposes up to eight spans. A shared scorer combines their log probabilities with projected contextual features of the passage, question and proposed answer. Each physical location occurs once. Probability is summed across locations with identical glyph images. The depth-one control uses the same features and architecture. Shuffling supervision within each question gave 91/359 exact replies. Neural reranking has also been used for text-based question answering. This experiment scores physical glyph spans and tests their effect on emitted replies. [Lean coverage and pooling](../formal/span_list/SpanList.lean) · [RankQA](https://aclanthology.org/P19-1611/)

The new scorer has 21,395 parameters. Computing shared features took 82.2 seconds and fitting all three heads took 20.1 seconds on one RTX 4090 D. These costs exclude inherited training of the Chinese reader and image writer. They establish feasibility of this adaptation. Independent language acquisition, connected book continuation, and a whole-system efficiency advantage remain open. [Training receipt](evidence/span_list_screened_20261124/fit.json) · [Reference and control code](../scripts/train_span_list.py) · [Portable CPU check](evidence/span_list_screened_20261124/release.json)

