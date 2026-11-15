# Whole-glyph ranking improves conditional writing but fails whitespace

The complete registered comparison improves readable nonempty glyphs while
failing its whitespace safeguard. All seven candidate groups exceed the fixed
90% readability requirement, with 1,002/1,040 selected images credited. False
blanks fall from 711 to 37 over all 12,597 supplied-history positions. However,
the candidate produces ink at **25/252 source-white positions**, exceeding the
unchanged control's maximum of **11/252**. The physical gate fails. No autonomous
answers or book continuations were generated for this actor, and its complete
language capability remains unmeasured.

## Complete comparison

The [registered protocol](global_glyph_ranking_training_protocol.md) is complete:
13,766 complete sources, 41,298 retained updates, 6,108,741 target exposures,
five fixed snapshots, the independent CPU audit and all 124 conditional cases.
The actual step-three restore and step-four/five replay agree in weights,
optimizer and RNG; two extra executed updates are counted separately. All 131
learned tensors are audited. The bound 91 sources and 421 inputs are preserved.
All model workers exited normally; this fit is terminal.

Each conditional state receives the supplied preceding source images. The
unchanged proper-likelihood actor supplies the control pixels; those pixels
were not regenerated. This measures conditional writing rather than autonomous
language or resilience to the model's own previous errors.

| Selected group | Ranking candidate | Same control pixels, fresh review |
|---|---:|---:|
| Development book, fixed | 106/108 | 90/108 |
| Development QA, fixed | 250/257 | 254/257 |
| Development QA, proportional | 255/257 | 255/257 |
| Training book, fixed | 100/108 | 97/108 |
| Training book, proportional | 98/108 | 95/108 |
| Training QA, fixed | 96/101 | 94/101 |
| Training QA, proportional | 97/101 | 96/101 |
| Total | 1,002/1,040 | 981/1,040 |

A credit means the reviewer can read a nonempty form without uncertainty. It
does not require the source character, correct semantics or a complete answer.
The control's original frozen review gave 970/1,040, including subsequently
identified review errors. That original score remains unchanged. The new
981/1,040 score is another review of identical control pixels, not a model gain
or a repair of the earlier review.

The following counts cover **every** position, including unselected images.

| Group | Positions | Source white | Candidate false ink | Candidate false blanks | Control false blanks |
|---|---:|---:|---:|---:|---:|
| Development book, fixed | 1,161 | 22 | 5 | 3 | 93 |
| Development QA, fixed | 745 | 2 | 1 | 0 | 11 |
| Development QA, proportional | 745 | 2 | 1 | 1 | 2 |
| Training book, fixed | 4,586 | 110 | 9 | 16 | 307 |
| Training book, proportional | 4,586 | 110 | 9 | 17 | 289 |
| Training QA, fixed | 387 | 3 | 0 | 0 | 8 |
| Training QA, proportional | 387 | 3 | 0 | 0 | 1 |
| Total | 12,597 | 252 | **25** | **37** | **711** |

Exact source bitmap agreement is 4,609/12,597 versus 4,603/12,597. Development-book
source-mask mean NLL increases from 0.0760705 to 0.0803382 nat despite improved
readable forms. Neither bitmap agreement nor likelihood alone measures language.
The ranking objective reduces the measured false-blank failure but does not
satisfy the complete prospective physical criterion. A shift toward nonempty
images is a possible explanation for the whitespace regression, not a cause
established by these counts.

## Review and reproducibility

All 33 opaque boards and 2,080 candidate/control images were actually inspected.
Empty flags were independently derived from original binary pixels. Readings
were frozen once before new owner/reference scoring; no credited candidate image
is empty. There are 28 uncertain readings and 69 actually empty reviewed images.
The reviewer was a model with prior development familiarity; independent human
adjudication has not been performed.

Reinspection found entry alignment errors in the initial **unfrozen** two-board
session. That complete session and its receipts were preserved. Corrections
used only opaque-output reinspection, before owner or reference access; strict
row-length and pixel-empty checks preceded recording the new complete session.
No frozen reading was repaired. The public export binds this correction receipt,
the first freeze, all image arrays and the failed gate.

The [CPU export](evidence/global_glyph_ranking_physical_result_20261115.json)
recounts both actors' complete arrays, verifies identical controls, reproduces
the fixed counts and authenticates reports, review and registration. It performs
no neural computation, optimization or new grading:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/summarize_global_glyph_ranking_result.py --verify
```

Fit execution took 77,431.213 seconds, including 1,681.590 seconds of pressure
waiting, at 18.281 GiB peak allocated VRAM and 45.736 GiB minimum available host
RAM. Independent CPU audit took 17.475 seconds at 1.131 GiB peak RSS. Conditional
qualification took 427.437 seconds at 18.035 GiB peak allocated VRAM. These costs
exclude inherited pretraining, earlier experiments and review. The whole model
has 7,553,448,370 parameters, including 7,522,463,984 inherited parameters and
5,005,760 trainables. This is not a compact foundation or a matched-quality
efficiency result.

The [next bounded decision](global_glyph_ranking_whitespace_decision.md) targets
this actor's new whitespace failure. Do not repeat the completed fit, choose an
intermediate checkpoint, repair frozen readings or launch its 708-stream
autonomous comparison. The existing 48/84/86-page Nature-style PDFs predate this
measurement and preserve the preceding failures. Their next results revision
must distinguish this conditional gain, failed whitespace and unmeasured
language; it must not replace failed samples with illustrative writing.
