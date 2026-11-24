# A pictured prefix influences classical Chinese book continuation

The writer reads 32 aligned glyph images and returns a strip of sixteen images. In the first passage from the *Analects* it generated 人不慍乎 four times. The phrase draws on the prompt, and the printed shapes are clear. The recorded continuation begins 有子曰其為人也孝弟. Across two completely reserved works, retaining the pictured context improved literal next-glyph recovery. Short cycles and incorrect continuations persisted.

| Writer | Correct with source history | Correct with self-feedback | Mean negative log probability on shared targets |
|---|---:|---:|---:|
| Image + pointer + suffix counts | 2,744 | 839 | 6.118 |
| Image + pointer | 2,613 | 969 | 6.248 |
| Suffix counts | 2,354 | 435 | 7.795 |
| Bytes + pointer + suffix counts | 2,809 | 886 | 6.172 |
| Bytes + pointer | 2,657 | 813 | 6.276 |

Every accuracy column covers the same 28,880 positions in 1,805 passages. Likelihood covers the 28,217 targets in the common fitting palette and uses natural logarithms. None of the five writers recovered any complete initial eight-glyph reference or any complete sixteen-glyph reference. Four initial eight-glyph targets and zero full sixteen-glyph targets appeared verbatim in fitting volumes. Literal scoring cannot judge other plausible continuations. [All position-level predictions and scores](evidence/book_prefix_v1/results.json)

## What the original prefix contributes

Keeping the last two glyphs and exchanging the earlier 30 with another book reduced the visual mixture's free recovery from 543 to 299 positions in the *Analects* and from 296 to 127 in *Chu Ci*. The pooled advantage was 1.43 percentage points (95% paired interval 1.04–1.95). Restricting the history at every step to its last two cells gave 634 correct overall. The paired intervals resample the 20 and nine volumes within their respective works. Cross-work exchange can change topic, style and verse structure simultaneously.

Given correct history, the image mixture gained 1.35 points over counts alone (0.80–2.03) and 0.45 points over the image reader with a pointer (0.12–0.78). Both registered lower bounds exceed zero. Its difference from the byte mixture was −0.23 points (−0.52 to 0.11). The count branch improved conditional prediction while reducing literal self-fed accuracy relative to the neural image reader. At the first held position 162 predictions matched. At the second, the mixture matched 187 with the true first glyph and 57 with its own first prediction. [Evaluation protocol](book_prefix_protocol.md)

## Output support and resources

For a palette and copy writer with its original prompt still visible, induction shows that a finite target trajectory is reachable exactly when every desired bitmap is in the palette or prompt. Here 532 positions fell outside that union. A further 27,509 errors occurred inside it. Nine of the 131 out-of-palette positions already pictured in the prompt were recovered. An expanded output palette alone could address only a small fraction of these misses. [Lean derivation](../formal/span_list/BookWriterSupport.lean) · [Independent checks and literal readbacks](evidence/book_prefix_v1/audit.json)

All four neural fits started from random weights, shared the same 3,200 window batches and scored 4,531,200 presentations each. The image mixture has 1,032,841 trainable parameters and took 20.16 seconds, 2,190 J of GPU energy and 150.3 MiB peak allocated PyTorch memory on device 0, an RTX 4090 D. Its release needs 4.14 MB of neural weights, 0.70 MB of palette and 1.93 MB of suffix counts (decimal). The byte mixture took 19.08 seconds and 2,100 J. These counters cover optimizer fitting on the board. Host data processing, package construction and inference have different costs. Timings under a common update count do not establish superior efficiency at equal language quality.

## Next language test

Determine whether the same final reader can distinguish two genuine endings when they follow the same recent glyphs in the same work. Score both assignments of two original passages to two observed continuations, then compare with the byte and suffix-only readers. This controls for a globally frequent ending and reduces the cross-work style cue. Treat this as a diagnostic of these fixed weights. For any subsequent learning comparison, reserve fresh sources before tuning. A successful passage assignment would still require a separate open-response writing test.

[Runnable image writer](../models/book_prefix/README.md) · [Data credit and provenance](https://github.com/UniversalDependencies/UD_Classical_Chinese-Kyoto/tree/r2.16)
