# Recognizing an authentic line and writing its missing phrase

A Tang-trained image writer assigned two supplied endings to their respective Song sources on 3,020 of 3,781 scorable comparisons (79.9%). Asked to write both missing glyphs, it restored eight of 3,854 phrases (0.21%). Its paired control restored 12. The four fits used one RTX 4090 D. These results set separate expectations for reading a supplied answer and writing one from the surviving evidence. [All measurements](evidence/verse_contrast_v1/results.json) · [Actual input and output](../models/verse_canvas/examples/li-liu-restored.png)

## What changed during training

Starting with the released 3,282,782-parameter ink-bound and byte-bound Tang models, each continuation trained for 4,000 steps of 64 line pairs. The same schedule exposed 145,021 distinct fitting pages in each arm. All arms learned ordered pair likelihood alongside masked writing and visible-glyph identification. The contrastive arms also compared authentic and exchanged assignments of two lines from the same poet and metre. Each source and each ending occurs on both sides. This use of logistic discrimination is related to contrastive estimation, although the writers here estimate normalized conditional glyph probabilities and receive direct writing supervision. [Protocol](verse_contrast_protocol.md) · [Gutmann and Hyvärinen, 2010](https://proceedings.mlr.press/v9/gutmann10a.html)

| Writer | Both cells under ordered width-eight search | Supplied reciprocal assignments | Fully blank lines recovered | Additional GPU energy |
|---|---:|---:|---:|---:|
| Image parent | 4 | 2,982 | 0 | — |
| Paired image | 12 | 2,970 | 0 | 7.88 kJ |
| Contrastive image | 8 | 3,020 | 0 | 11.77 kJ |
| Byte parent | 11 | 2,972 | 0 | — |
| Paired byte | 14 | 2,941 | 0 | 8.51 kJ |
| Contrastive byte | 17 | 3,012 | 0 | 14.09 kJ |

Eight fixed shards yielded 3,854 Song pages from 192 distinct poet names. The measured pages shared neither source nor target line bitmaps with the retained Tang partitions. Names are corpus attributions. Printed test shapes outside the training bank remained in the test. For 3,768 source exchanges the donor was another line of the same poet. All original pages entered the writing denominator. Reciprocal assignment required the four hidden reference cells across the two pages to be representable. [Frozen corpus](evidence/verse_contrast_v1/corpus.json) · [Chinese Poetry corpus](https://github.com/chinese-poetry/chinese-poetry)

All eight contrastive image successes were shared with the paired image control, which also completed four other repairs. The difference was −0.10 percentage points (95% paired poet-bootstrap interval −0.22 to 0.00). The model recovered 266 of 7,708 individual missing glyphs, compared with 256 under the paired control. With a fully blank line it matched 688 of 23,988 positions, below aligned fitting counts (773). The eight ordered repairs dropped to two when the first line was exchanged. Literal agreement admits one reading and does not score acceptable alternatives. [Scores, intervals and denominators](evidence/verse_contrast_v1/results.json)

## Where the recorded pair was lost

The image writer placed the correct first glyph in its top eight on 555 pages. With the true first glyph supplied, the correct second glyph scored highest on only 151. Both conditions held on 28 pages, counting rank ties favourably. Wider first-cell search alone therefore cannot recover more than 151 recorded pairs under this second-cell rule. Only 38 pages contained an unrepresentable reference pair. The current width-eight prediction had a certificate of global optimality under its own distribution on 82 pages. The Lean proof uses the largest omitted first-glyph score as an upper bound on every excluded pair. This certifies a decoding property; language quality is measured by actually writing. [Lean derivation](../formal/span_list/OrderedGlyphPair.lean) · [Scorer and decoder](../ilm/visual_lm/verse_phrase.py)

On the first held page the damaged text was 新□□憶共清尊. The writer supplied 人不 where the recorded line has 涼殊. On the first exact repair lost after source exchange it supplied 清風 in 李劉's 故揖□□作頌聲. The public PNG interface reproduced both measured outputs while preserving every supplied cell. No LLM, OCR or Unicode decoding is required when running from the two-row page. [Image interface](../models/verse_canvas/README.md)

The contrastive image continuation took 111.5 seconds and 11.77 kJ of device energy. Its parent took 87.3 seconds and 10.14 kJ. Their summed GPU energy is 21.91 kJ, excluding corpus preparation, evaluation and host power. Equal examples required more compute for the additional contrast term. No equal-quality efficiency advantage was established. [Parent](evidence/verse_canvas_v1/ink-bound-fit.json) · [Continuation](evidence/verse_contrast_v1/ink-contrast-fit.json)

The original writing default remains available. A test closer to answering an etymology question is to read an entry from a larger illustrated collection, find the evidence for the requested headword and write its attested forms in the requested order. Prior source-plate tests supplied the correct four-row plate. A fresh comparison should withhold that location, use headword-family splits, and score retrieval, both stage images and stopping end to end. This tests whether a user can ask a useful question about a book. Open-ended historical explanations and generating unseen forms without a source require separate evidence. [Prior supplied-plate result](historical_plate_result.md)
