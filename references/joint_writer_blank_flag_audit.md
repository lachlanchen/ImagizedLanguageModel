# Pixel audit detects errors in the frozen physical review

A CPU-only audit of all 2,080 selected images finds **17 disagreements between
the frozen blank flag and actual emitted ink**. Four empty images received
readability credit: three from the joint writer and one from the point control.
Thirteen nonempty images were marked blank. Every original reviewed PNG matches
its authenticated saved array. The inconsistency is between the review and
the pixels, rather than a changed source image.

| Original review group | Blank-flag disagreements | Credited empty images | Nonempty images marked blank |
|---|---:|---:|---:|
| Point control, training QA, proportional | 1 | 0 | 1 |
| Point control, training books, fixed | 1 | 0 | 1 |
| Point control, training books, proportional | 2 | 1 | 1 |
| Joint writer, training books, fixed | 1 | 0 | 1 |
| Joint writer, training books, proportional | 1 | 0 | 1 |
| Joint writer, development QA, fixed | 2 | 1 | 1 |
| Point control, development QA, fixed | 3 | 0 | 3 |
| Joint writer, development books, fixed | 4 | 2 | 2 |
| Joint writer, development QA, proportional | 2 | 0 | 2 |
| Total | 17 | 4 | 13 |

The original frozen grades remain 970/1,040 candidate credits and 599/1,040
control credits, with two candidate groups below the unchanged 90% threshold.
These are retained review counts, not corrected physical accuracy. An inked
image is not automatically readable. No new character transcription, corrected
score or successful language result is inferred from this binary audit.

The development-book group illustrates why marginal counts are insufficient.
There are seventeen uncredited images and seventeen pixel-empty images, but
only fifteen images belong to both sets. Two nonempty images were marked blank;
two other empty images were transcribed and credited. Among the 108 selected
source positions, four source images are blank, three overlapping empty outputs.
The number of false blanks is therefore fourteen, matching the public
probability JSON. Earlier prose incorrectly subtracted all four source blanks
from seventeen output blanks and reported thirteen.

The [source-bound audit](evidence/joint_writer_blank_flag_audit_20261114.json)
retains every discrepant opaque ID, case, position, original reading and source
array hash. The figure extractor reproduces it without a model call:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  python publication/ilm-nature/prepare_joint_writer_figure.py
```

Original arrays, readings, uncertainty flags, registrations and verdicts remain
unchanged. The head-only trial still fails its prerequisite and receives no
autonomous language score. Its single-model-reviewer limitation is material.
Future physical qualification needs both frozen visual readings and an
independent check that credited images contain ink; a score alone does not
prove the intended readability requirement. The current context-gradient fit
continues under its original registration and resource limits.
