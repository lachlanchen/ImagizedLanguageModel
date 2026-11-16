# White-target ranking reduces false ink but fails the physical prerequisite

The complete white-target comparison fails both registered blank conditions.
It produces **12 false-ink outputs among 252 true-white positions**, above the
maximum of eleven, and **54 false blanks**, above the maximum of 37. The unchanged
nonempty-ranking control gives 25 and 37, respectively. All seven readability
groups pass, with 1,016/1,040 selected candidate images credited. This conditional
writing result does not qualify the actor for autonomous language evaluation.

## Complete comparison

The [fixed protocol](white_target_training_protocol.md) is complete: 13,766
sources, 41,298 retained updates, 6,108,741 scheduled target positions, all five
snapshots, the independent CPU audit and all 124 conditional cases. The audit
recounts 68,808 source-white target exposures and all 131 trainable tensors at
each fixed snapshot. Two discarded replay updates are counted separately.
All 115 bound sources remain unchanged. The original serial workers exit normally.

Every conditional state receives the correct preceding source images. Candidate
and control outputs have identical complete source arrays; the control pixels
are reused from the completed nonempty-ranking actor and are not regenerated.
These measurements assess supplied-history writing rather than autonomous
answers, question dependence, coherent continuation or natural stopping.

| Selected group | White-target candidate | Same control pixels, fresh review |
|---|---:|---:|
| Development book, fixed | 106/108 | 106/108 |
| Development QA, fixed | 256/257 | 253/257 |
| Development QA, proportional | 253/257 | 255/257 |
| Training book, fixed | 102/108 | 102/108 |
| Training book, proportional | 102/108 | 100/108 |
| Training QA, fixed | 99/101 | 97/101 |
| Training QA, proportional | 98/101 | 97/101 |
| Total | 1,016/1,040 | 1,010/1,040 |

A credit denotes a readable nonempty form, not the correct source character or
a correct answer. The control's earlier frozen review gave 1,002/1,040. The new
1,010/1,040 is another review of the same pixels; it neither changes that original
score nor establishes a model improvement.

The following counts cover every position, including unselected images.

| Group | Positions | Source white | Candidate false ink | Candidate false blanks | Control false blanks |
|---|---:|---:|---:|---:|---:|
| Development book, fixed | 1,161 | 22 | 5 | 9 | 3 |
| Development QA, fixed | 745 | 2 | 0 | 0 | 0 |
| Development QA, proportional | 745 | 2 | 0 | 1 | 1 |
| Training book, fixed | 4,586 | 110 | 3 | 21 | 16 |
| Training book, proportional | 4,586 | 110 | 3 | 23 | 17 |
| Training QA, fixed | 387 | 3 | 1 | 0 | 0 |
| Training QA, proportional | 387 | 3 | 0 | 0 | 0 |
| Total | 12,597 | 252 | **12** | **54** | **37** |

The additional seventeen false blanks all occur in books. Exact source bitmap
agreement falls from 4,609 to 4,543 of 12,597. Development-book source-mask mean
NLL decreases from 0.0803382 to 0.0788611 nat while false blanks increase from
three to nine and exact bitmaps decrease from 222 to 202. Thus this loss change
improves some saved probability summaries without satisfying the complete physical
criterion. These counts identify a measured tradeoff, not its unique cause.
Bitmap identity is also stricter than readable-character identity and is not a
language score.

## Review and reproduction

All 33 opaque boards and 2,080 candidate/control images were actually inspected.
Nearest-neighbor enlargements preserve the original pixels. Eighteen readings
remain uncertain and 36 reviewed tiles are physically empty. Blank flags are
derived from original pixels, and no credited candidate image is empty. Readings
were frozen once before new owner/reference scoring and remain unchanged.

Two incomplete unfrozen submissions were rejected by pixel blank validation.
The original entries and failures are preserved. Replacements used only opaque
reinspection before board recording and the global freeze. Review uses one model
reviewer with prior development familiarity; there is no independent human
adjudication. Canonical mismatch alone does not establish semantic falsity.

The [public aggregate evidence](evidence/white_target_physical_result_20261116.json)
binds the original reports, review seal, source registration and exporter.
The CPU exporter independently recounts both actors' complete arrays and checks
the unchanged controls. It performs no neural computation, optimization or new
grading:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/summarize_white_target_result.py --verify
```

Fit execution takes 77,076.359 seconds, including 1,461.393 seconds of pressure
waiting, at 18.281 GiB peak allocated VRAM and 53.312 GiB minimum available host
RAM. CPU audit takes 15.474 seconds at 1.170 GiB peak RSS. Qualification takes
382.482 seconds at 18.035 GiB peak allocated VRAM, with no pressure waiting.
Physical gate scoring takes 1.372 seconds at 0.609 GiB peak RSS. These measurements
exclude inherited pretraining, prior experiments, source preparation and visual
review. The whole deployed model has 7,553,448,370 parameters, including
7,522,463,984 inherited parameters and 5,005,760 trainables. They establish neither
a compact foundation nor matched-quality efficiency over a local LLM.

The [failure decision](white_target_failure_decision.md) stops this actor before
its 708-stream autonomous comparison. Language quality remains unmeasured.
The current Nature-style reading/review/redline PDFs now include this completed
tradeoff and native Figure 30, with original source/control/candidate tiles and
all earlier evidence preserved. The previous PDFs remain baseline artifacts.
The [revision receipt](../publication/ilm-nature/evidence/visual_context_evidence_article_build.json)
separates this physical failure from the subsequently registered full-context
fit, whose complete language remains unmeasured.
