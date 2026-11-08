# Complete native full-page QA assessment

Numerical assessment and raw visual inspection of all384 answer pages are complete. No manual transcription or answer-score replacement is claimed. No reliable language or writing capability is qualified.

| Source-accepted OCR observation | Initial | Final |
| --- | ---: | ---: |
| Seen answers | 0/32 | 0/32 |
| Held-out answers | 0/32 | 0/32 |
| Held-out both-correct passage pairs | 0/16 | 0/16 |
| Held out, question erased | 0/32 | 0/32 |
| Held out, passage erased | 0/32 | 0/32 |

| Canonical OCR copying observation | Initial | Final |
| --- | ---: | ---: |
| train | 18/32 | 19/32 |
| validation | 19/32 | 16/32 |

The prospective exploratory conditions are recorded individually in [result.json](result.json). They do not establish broad language competence, reliable writing or efficiency at matched quality. The source preparation uses public development data already examined in earlier work. Canonical-reference OCR is imperfect: 23/32 seen and 26/32 held-out answers. The first 32 natural questions are optimized for four complete passes; only final update 128 is assessed. Initial weights are fixed before QA training and assessed afterwards with the same inputs and seeds.

All 384 initial/final answer images and 128 actual pre-update training images are retained, with hashes. There are 320 new native assessment calls and 64 reused initial copying outputs. Every original input is preserved; only contact-sheet input previews are resized to fit. Every displayed model answer is complete and unrepaired. Galleries are listed in result.json; source reports with local filesystem paths made relative and independent training verification accompany them. Original report hashes and public report hashes are retained separately.

The runtime retains the externally pretrained transformer and VAE. Training-only OCR supervision is excluded from the standalone image generator. Parameter counts, costs, source hashes and the fixed generic condition are in [training verification](training_verification.json) and [training report](reports/training_report.json). [Dataset attribution and transformations](SOURCE_ATTRIBUTION.md), [original dataset licence](CMRC2018_LICENCE).

[Raw visual-review coverage](raw_visual_review.json) binds all384 answer cases.
