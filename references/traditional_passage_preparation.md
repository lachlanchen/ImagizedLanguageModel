# Traditional Chinese passage evidence

The Delta Reading Comprehension Dataset adds human-authored traditional Chinese questions with attested answer spans. The prepared set contains 21,934 training questions on 6,579 passages and 3,505 development questions on 994 passages. The training passages contain 2,865,530 pictured glyph cells. This is a prepared corpus. No DRCD fit is included in the reported CMRC results. [DRCD upstream data and terms](https://github.com/DRCKnowledgeTeam/DRCD/tree/b944790de5af02c5fbb7cd9cb1473d27d169eebf) · [Verified counts and hashes](evidence/drcd_passage_preparation_20261119.json)

The original glyphs are retained. Simplified conversion and Unicode normalization serve only to screen for overlapping texts. The training split excludes titles or matching passages from the official DRCD development and test releases and both CMRC releases. The development split excludes CMRC overlaps. The procedure also compares every candidate 32-character window with aligned 32-character windows of the reserved documents. Any common contiguous normalized run of 63 characters necessarily includes such a window. Shorter paraphrases, translations and every possible version of a shared article are outside this check. Within each prepared split the procedure excludes duplicate complete normalized passages.

Of 26,936 original training questions, 4,866 were excluded by overlap screening and 136 required an unavailable passage glyph. Of 3,524 development questions, 19 required an unavailable passage glyph. The physical archives retain all aligned occurrences of all accepted answers. An independent readback compared the actual packed passage and question bitmaps and recounted all 42,339 training intervals and all 7,687 development intervals against the upstream records.

The local prepared corpus is in `data/traditional_passage_evidence_v1/{train,development}/`. Each directory contains `pixels.npy` (packed 32 × 32 glyphs), two offset-indexed passage and question arrays, original question identifiers and `evidence.npz`. Its group array identifies the passage for each question, and its offsets delimit half-open answer intervals in that passage. The training reader can marginalize over multiple correct occurrences, and contrast each question's evidence with the answers to other questions about the same passage.

The upstream snapshot belongs in `data/raw/drcd/`. To reproduce it, fetch the three JSON releases and README at commit `b944790de5af02c5fbb7cd9cb1473d27d169eebf` and write `origin.json` using the exact `preparation.upstream` object from the verification report. Run the following from the repository root with the CMRC source files and project font available.

```bash
PYTHONPATH=. python scripts/prepare_drcd_passage_ink.py
PYTHONPATH=. python scripts/audit_drcd_passage_ink.py
```

The preparation creates a new dataset directory and preserves previous outputs. The independent audit verifies the committed receipt on replay. The source records were compiled and adapted from Wikipedia by the Delta Research Center and released under CC BY-SA 3.0. The local derived corpus retains these terms. [DRCD copyright notice](https://github.com/DRCKnowledgeTeam/DRCD#copyright-notice-%E7%89%88%E6%AC%8A%E8%81%B2%E6%98%8E)
