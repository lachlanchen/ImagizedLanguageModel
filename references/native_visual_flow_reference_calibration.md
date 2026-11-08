# Reference-page OCR for native visual flow learning

The prepared native flow experiment uses 1024-square answer pages, preserving
the original 512-by-256 answer region and adding only white padding. Its
offline evaluator has now scored every reference page in the four fixed
assessment cohorts. No candidate model, codec or learned output is evaluated
by this calibration.

| Reference cohort | Seen, OCR exact | Held out, OCR exact |
| --- | ---: | ---: |
| Normal Chinese QA | 23/32 | 26/32 |
| Question erased, Chinese QA | — | 26/32 |
| Passage erased, Chinese QA | — | 26/32 |
| Chinese/English copying | 27/32 | 29/32 |

The erasure cohorts reuse the normal held-out answer pages. There are 192
reference observations across the four cohorts, covering 128 normal QA and
copying cases. The [calibration record](evidence/native_visual_flow_reference_calibration.json)
binds every report to its prepared reference manifest, OCR executable,
recognition data and evaluation implementation. The separate generator
input indices contain no reference text or target image.

Imperfect recognition of clean references limits the interpretation of
generated-image exact-match scores. Retain source-accepted QA alternatives,
uncertain readings and complete raw-image inspection. Reference calibration
does not validate recognition of malformed generated typography, and does
not supply evidence of semantic learning. Actual candidate reconstruction,
generation, backward and memory checks remain pending.
