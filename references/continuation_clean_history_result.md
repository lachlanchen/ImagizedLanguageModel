# Clean history improves later QA writing but does not solve books

The [registered comparison](continuation_clean_history_protocol.md) is complete:
124 complete cases at each initial/final checkpoint, 248 traces and 25,194 target
positions. Targets come from authenticated corpus pixels. Correct prior images
and target lengths are supplied; stopping is bypassed. These are assisted physical
predictions, not autonomous answers or new semantic grades. Parameters remain
unchanged.

| Final supplied-history group | Cases | First-image F1, case mean | Later F1, position mean | Later fraction F1 ≥.95 | Tiles above clean ink 99th percentile |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training QA, fixed | 12 | .6647 | .8661 | 65.87% | 22.48% |
| Training QA, proportional | 12 | .7739 | .9905 | 95.47% | 5.43% |
| Development QA, fixed | 32 | .5508 | .7613 | 40.81% | 41.34% |
| Development QA, proportional | 32 | .6942 | .9827 | 92.01% | 5.77% |
| Training books, fixed | 12 | .5309 | .6025 | 9.03% | 75.32% |
| Training books, proportional | 12 | .5395 | .6022 | 9.01% | 75.34% |
| Development books, fixed | 12 | .3231 | .5587 | 4.61% | 81.91% |

All initial groups and per-case results remain in the complete
[evidence](evidence/continuation_clean_history_result_20261113.json). Development
QA has 745 target positions per layout, including 713 later positions. The final
proportional group has 656/713 later positions at F1 .95 and 512 exact bitmaps,
compared with 374/713 and 131 exact bitmaps initially. Training QA has 387 positions
per layout, including 375 later positions; each training-book layout has 4,586
positions and development books 1,161. Short targets and punctuation remain.

The complete 513-glyph parallel/incremental check passes at both checkpoints:
maximum state errors are .000534058 initially and .000148773 finally, below .002.
Every original development observer geometry is reproduced; maximum first-state
difference is .000120163. This is not bit-exact shape-independent arithmetic.

Both proportional QA groups pass the registered descriptive 90% later-writing
condition. Their first-image and autonomous failures remain. Books fail even on
sampled training records, rejecting a feedback-only explanation. No book group
meets the separate greater-than-90% dense-description rule: report the measured
75–82% excess-ink fractions directly. Supplied history can make later glyphs easier
to predict; these results do not identify a unique replacement algorithm.

All twelve development books' reference, supplied-history and autonomous first/
last-eight images were actually inspected. Some forms and punctuation recover
under correct history, but many tiles remain dense. Different stream lengths
prevent semantic alignment of their tails. No output is repaired or regraded.

The successful stage takes 648.971 seconds, including 30.029 seconds of pressure
waiting, at 18.124 GiB peak allocation and at least 49.819 GiB available host RAM.
An earlier launch fails before weight loading because the CUDA workspace setting
was absent. Its .341-second measured stage and unknown whole-process overhead
are preserved in the [recovery receipt](evidence/continuation_clean_history_launch_recovery_20261113.json).
No model worker, dedicated desktop or GPU0 allocation remains afterward.

The [selected joint writer](continuation_joint_writer_decision.md) preserves the
actual final reader/context and replaces the marginal point writer. Its CPU
tests are not language evidence. Full-corpus preparation, fit registration and
complete autonomous evaluation remain required. The full ILM goal remains open.
