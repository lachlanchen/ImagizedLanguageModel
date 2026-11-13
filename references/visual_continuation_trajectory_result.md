# Dense-image failure in the complete continuation trial

All 732 saved streams and 416 first-step contrasts are measured without new
model calls. The analysis uses every eligible complete training target: 13,766
sequences and 2,036,247 glyph positions. Outputs and whole-answer readings remain
unchanged. The [registered analysis](visual_continuation_trajectory_analysis_plan.md)
precedes these statistics, but follows the completed language scoring.

The constant-image optimum of the actual sequence-balanced BCE has 504 black
pixels out of 1,024 after the runtime threshold, an ink fraction of 49.219%.
Its balanced BCE is 0.437380. This is an optimum only among context-independent
images, not among conditional neural models. Clean QA and book glyphs have
median ink counts of 188 and 201; their 99th percentiles are 267 and 276.

| Normal autonomous arm | Above clean 99th percentile | Early ink | Tail ink | Early template F1 | Tail template F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial proportional QA | 26.04% | .2564 | .2328 | .6583 | .6169 |
| Final proportional QA | 86.32% | .4378 | .4725 | .9137 | .9336 |
| Initial fixed 640-cap books | 5.94% | .2113 | .2335 | .5599 | .6220 |
| Final fixed 640-cap books | 98.12% | .4375 | .4718 | .8990 | .9386 |

The percentile column pools emitted tiles. Early and tail columns first average
the first or last eight tiles within each stream, then average streams with at
least 32 emitted tiles. These disjoint-window populations contain respectively
20, 9, 4 and 12 streams; they must not be interpreted as matched longitudinal
comparisons of identical subsets. Short answers are retained in the complete
[evidence](evidence/visual_continuation_trajectories_20261113.json).

All twelve final normal long-book streams were inspected again as a diagnostic
plate, showing their first and last eight actual emitted tiles. Dense blocks
are visible from the beginning and persist later. This inspection does not
regrade the frozen complete-answer review. The result supports a description
of dense, similar writing; it neither proves a dynamical attractor nor identifies
the training loss as the sole cause.

Every one of the 416 first-step contrasts has unequal contextual states; all
413 contrasts in which both streams emit also have unequal first images.
At the final proportional checkpoint, paired-question first-state relative L2
averages 1.0428 and first-image Hamming distance averages .2302. Representation
change therefore persists despite failed paired language answers. Distances
alone do not establish question understanding or correct passage use.

The successful analysis takes 18.879 seconds including its wrapper, with
0.593 GiB peak RSS and at least 51.210 GiB available host memory. An earlier
attempt returned SIGTERM/143 without a report; its cause is unknown and its
wall-time upper bound is 242.049 seconds. Exact cumulative execution cost is
unknown. Neither attempt invokes a model or changes an optimizer.

The next unresolved contrast is the new final checkpoint under supplied correct
history, against its unchanged initial checkpoint. If dense images already occur
on clean histories, feedback-only repairs are insufficient for this checkpoint.
If clean-history writing succeeds but autonomous writing fails, generated-history
training or representation remains a plausible target. A new bounded protocol
must also check full-length parallel/incremental agreement before interpreting
this contrast. Earlier clean-history results used a different actor.

Related pixel-language work already studies blurry generation and free-running
refinement: [PIXAR](https://arxiv.org/abs/2401.03321) and
[MIXAR](https://arxiv.org/abs/2604.11575). Their results do not establish the cause
of this failure or single-4090 efficiency. Earlier local joint-pixel and flow
writers also failed language gates; a different sampler alone is not a supported
repair.

Reproduce the public export without invoking neural inference:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/export_continuation_trajectory_result.py --verify
```
