# Context learning lowers image likelihood loss but increases book blanks

The complete context-gradient comparison fails its registered physical-writing
prerequisite. All 41,298 retained updates, five snapshots, the independent CPU
audit and 124 supplied-history qualification cases are complete. All 33 opaque
boards were viewed; 2,080 first readings were frozen before opening the owner
mapping. The candidate misses the unchanged 90% readable-nonblank requirement
in all three book groups. Its autonomous language quality remains **unmeasured**.
The failed actor does not proceed to the planned 708-stream language evaluation.

| Supplied-history group | Context-gradient candidate | Unchanged head-only pixels, freshly reviewed |
|---|---:|---:|
| Development books, fixed | 93/108 (86.11%) | 90/108 (83.33%) |
| Development QA, fixed | 250/257 (97.28%) | 245/257 (95.33%) |
| Development QA, proportional | 251/257 (97.67%) | 249/257 (96.89%) |
| Training books, fixed | 95/108 (87.96%) | 103/108 (95.37%) |
| Training books, proportional | 92/108 (85.19%) | 101/108 (93.52%) |
| Training QA, fixed | 94/101 (93.07%) | 92/101 (91.09%) |
| Training QA, proportional | 95/101 (94.06%) | 98/101 (97.03%) |

These are original frozen **review credits**, not corrected physical accuracy.
The independent pixel audit finds 28 blank-flag inconsistencies, including four
candidate and two control empty images credited as readable. Original readings,
uncertainty, grades and the failed gate are preserved. The independent
credited-ink prerequisite also fails. Review used one model reviewer with prior
cohort familiarity and no independent human adjudication. Different frozen
readings of the same control pixels must not be presented as a change in that
control model. Canonical glyph agreement and complete language are separate
measurements.

## Lower source loss does not qualify book writing

The candidate trains 2,363,392 continuation parameters and 2,642,368 writer
parameters through the complete physical-image likelihood. Its native body,
prompt memory, pixel-feedback encoder, stop parameters and affine buffers stay
frozen. The deployed model contains 7,553,448,370 unique parameters, including
7,522,463,984 inherited parameters. A 5,005,760-parameter trainable interface
does not establish a compact foundation model.

All 13,766 complete sources receive three scheduled updates: one fixed-cell and
two proportional presentations, for 6,108,741 target exposures. The independent
audit authenticates the complete ledger, five actual snapshots, optimizer states
and exact saved step-three restore/replay. Two additional replay updates are
counted separately from retained exposure. Context states were recomputed with
gradients on every update; cached predictor states did not replace the graph.

The paired ledger comparison joins every scheduled update with the completed
head-only control by update, source index, source ID, layout and complete target
length. In the third epoch, all 4,762 book presentations have lower online source
image loss. Source-mean mask NLL changes from 0.09268 to 0.06639 for fixed books
and from 0.09209 to 0.06594 for proportional books. These are evolving training
states and teacher-forced physical-pattern losses, not autonomous checkpoint
scores or evidence of book understanding.

The final, supplied-history qualification shows the same likelihood/output
disconnection on the original complete arrays:

| Book group | Complete positions | Source-mask NLL, control → candidate | False blank outputs, control → candidate | Source-pixel-exact outputs, control → candidate |
|---|---:|---:|---:|---:|
| Development, fixed | 1,161 | 0.10716 → 0.07607 | 42 → 93 | 236 → 227 |
| Training, fixed | 4,586 | 0.07653 → 0.05528 | 137 → 307 | 1,331 → 1,291 |
| Training, proportional | 4,586 | 0.07653 → 0.05531 | 145 → 289 | 1,327 → 1,295 |

A false blank here means an empty generated tile at a nonempty source position.
Legitimate source whitespace is counted separately: 22, 110 and 110 positions
in these three groups. Across all 124 cases and 12,597 complete positions,
false blanks increase from 349 to 711; book false blanks increase from 324 to
689. QA source-pixel agreement improves in each layout, while the three book
groups lose exact reconstructions. These post-freeze pixel measurements do not
repair the visual review, rescore the gate or assign autonomous language scores.

This result rejects the tested gradient-reach change as a sufficient repair.
It does not identify a unique cause or disprove visual language learning.
Gold physical-prefix likelihood can improve while free greedy mask decoding
becomes less useful. The current data do not yet establish whether a better
whole-glyph search would recover the missed source forms, or whether the learned
density itself favors empty writing. Repeating this fit, choosing an intermediate
snapshot, changing whitespace rules or lowering the threshold is not a remedy.
The [next bounded decision](joint_context_failure_decision.md) addresses that
remaining distinction on this actor without rerunning the old control.

## Resources and reproducibility

Coordinator-measured child execution is 76,158.268 seconds for fitting, 15.255
seconds for the CPU audit and 477.432 seconds for qualification. The fit worker
reports 76,155.600 seconds, including 1,701.612 seconds of pressure waiting,
at 18.281 GiB peak allocated VRAM and 37.694 GiB minimum available host RAM.
Qualification reports 476.074 seconds, 40.037 seconds of pressure waiting and
18.035 GiB peak allocated VRAM. Prior reader training, preparation, inherited
pretraining and review work are separate costs. These values establish one-GPU
execution of this comparison, not matched-quality efficiency over a local LLM.
All original pipeline workers have exited; no automatic retry was launched.

The [source-bound export](evidence/joint_context_physical_result_20261114.json)
binds the reports, snapshots, ledgers, control pixels, review and ink audit.
The CPU exporter verifies all 59 registered sources, all complete qualification
arrays and every matched training update without a neural call. Its first run
took 3.565 seconds at 0.433 GiB peak RSS. Reproduce its deterministic contents:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/summarize_joint_context_result.py --verify
```

The existing 45/80/82-page manuscript PDFs predate this completed measurement.
They retain the earlier failures and current method; they do not yet contain
this result. The next bounded manuscript revision must include the loss/output
disconnection, original review limitations and actual samples in native figures.
The full language, independence, historical-form and efficiency goal stays active.
