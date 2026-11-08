# Paired image questions available for the next language assessment

The existing CMRC-derived page set has usable source bindings for a subsequent
reading experiment. A complete CPU audit verifies all **448 prepared pairs
and 896 image files** against the retained text source, manifest, text
checksums and renderer identity. All measured ink boxes are inside their
canvases. This establishes data bindings and edge geometry, not successful
reading through the neural model.

| Natural-question partition | Training | Held out from this optimizer |
| --- | ---: | ---: |
| Questions | 256 | 32 |
| Same-passage question pairs | 128 | 16 |
| Articles | 128 | 16 |
| Target literally appears in its question | 0 | 0 |
| Paired accepted-answer sets overlap | 0 | 0 |

Articles are disjoint across these partitions. Each pair has the same full
passage, different questions and different accepted answers. Training passages
contain 265–456 displayed characters and targets contain 1–16; held-out
passages contain 281–425 and targets contain 2–16. The question targets occur
in their source passages. The set also contains 128 training and 32 held-out
copying examples. None of the natural questions is used in the current
balanced copying continuation.

The first two held-out question images were inspected at readable size. They
share a passage about 四线天竺鲷 and ask for its distribution and type locality.
The expected answers differ: 印度西太平洋区 and 印度Pondicherry. The prepared
images retain the passage and question labels, with unfilled lower page area.
Inspection of these two images does not establish legibility of every page
or information retention by the frozen neural image encoder.

The input canvas is 1024x1024 at 32-pixel Noto Sans CJK; the answer canvas is
512x256. This differs substantially from the current short copying inputs.
Before training on it, a real native preflight must measure memory, execution
and representation at this geometry. Do not infer fit or reading quality
from the copying run or an image checksum. Preserve complete passages and
hold question/target information out of deployed helper code.

A later semantic test must score complete answers to both questions about
each unseen passage and include question/passage controls. Legibility,
answer correctness and dependence on the question remain separate outcomes.
This public development data has informed earlier experiments; it is not a
new blind generalization set. Passing a narrow extractive-reading experiment
would still leave the wider independent glyph-language goal open.

The [complete audit](evidence/paired_language_data_readiness.json) retains
every source ID, pair, measured ink box and shortcut check. It makes zero
neural calls. Reproduce from the repository root using the existing local
prepared data:

```bash
PYTHONPATH=. python scripts/audit_paired_language_data.py \
  --pages artifacts/image_edit_learning/pages_v2 \
  --source data/visual_qa/cmrc2018_short/train.jsonl \
  --output artifacts/paired_language_data_audit.json
```

This command checks neither OCR legibility, defects inside glyphs, neural
memory fit nor semantic understanding. It does not change the active training
schedule, source pages, language requirements or current failed qualification.

## Prepared native controls

The [complete control preparation](evidence/paired_language_controls_preparation.json)
now reproduces all 288 normal question pages exactly and prepares all 32
held-out question-erased and passage-erased inputs. Every passage pixel is
retained in question erasure; every question pixel is retained in passage
erasure. The 16 question-erased pairs have identical input pixels and seeds.
Their constant answer-label position is normalized within each pair, so
question length cannot remain as a pair-specific cue. This label-position
intervention is disclosed in the [native preflight protocol](paired_language_native_preflight_protocol.md).
The first held-out pair's two control types were visually inspected.

[Independent CPU reference OCR](evidence/paired_language_reference_ocr.json)
recognizes 23/32 seen and 26/32 held-out clean answer images. This is a
calibration limitation, not a neural model score. Some missed pages are
readable Chinese strings; punctuation and a full-width percent sign also
cause exact-match disagreements. Preserve these failures and require raw
answer review rather than selecting expected transcripts.

Reproduction uses `scripts/prepare_paired_language_controls.py`; its outputs
remain in the ignored `artifacts/paired_language_controls_v1/` directory.
The current ink-mass copying comparison continues unchanged. The subsequent [native preflight](native_paired_language_preflight_result.md)
now completes four source-page reconstructions, two real discarded updates and
eight native generations. The fixed earlier checkpoint answers0/4normal
questions correctly; this does not establish language learning.
