# Conditional writing from retained contextual states

The final motor writes many selected training forms recognizably, but fails
both preregistered physical-writing gates. The diagnostic makes no optimizer
update or contextual-body call. It generates one glyph from each of 32 first
and 32 later training states, selected by a committed hash order, and reuses
the retained first images from all 64 normal development inputs.

| 32 training positions per pool | Candidate median F1 | Candidate F1 ≥0.95 | Candidate exact bitmap | Parent median F1 | Parent exact bitmap |
|---|---:|---:|---:|---:|---:|
| First | 0.986408 | 32/32 | 4/32 | 0.990277 | 13/32 |
| Later | 0.974518 | 23/32 | 0/32 | 0.997523 | 15/32 |

Each gate requires median foreground F1 at least 0.99 and at least 30/32
samples with F1 at least 0.95. First-position mean F1 improves from 0.872790
for the parent to 0.984237 for the candidate, reducing several large writing
errors. Later mean F1 changes from 0.963619 to 0.954535. These are small fixed
training-state samples, not population or language-accuracy estimates.
Direct supplied-target reconstruction is 31/32 exact for first positions
and 32/32 for later positions. Its mean F1 is 0.999932 and 1.0 respectively.
Thus the same decoder can reproduce these supplied forms more accurately.

One later nonterminal training state predicts stopping with probability
0.99542. This diagnostic explicitly writes one glyph regardless of stopping
and reports that decision separately; it is not an autonomous answer. No
first training state predicts stopping. Later feedback cannot explain errors
made when starting directly from these retained correct-history states.

| Canonical development first forms, 32 per layout | Candidate mean F1 | Parent mean F1 | Candidate F1 ≥0.95 | Candidate exact bitmap |
|---|---:|---:|---:|---:|
| Fixed cells | 0.447771 | 0.492173 | 1/32 | 0/32 |
| Proportional | 0.661558 | 0.632918 | 9/32 | 0/32 |

These images are reused without resampling. The adapted native helper begins
with the canonical first character on 5/32 fixed and 18/32 proportional
questions. In those subgroups, candidate mean F1 is 0.770103 and 0.817164;
1/5 and 8/18 reach 0.95. The helper remains outside the deployed model.
Canonical bitmap mismatch is not character or semantic error: alternative
wording, learned stroke variations and helper wording differences remain
possible. Raw boards show many changed or damaged forms before any generated
history is fed back. The failure therefore cannot be assigned entirely to
accumulated feedback.

The conditional cache contains 870 unique first-target forms, 2,141 later
forms and 2,245 combined forms. This differs from the codec's 4,087 distinct
training bitmaps. Six of the 32 canonical first positions have no first-pool
example; only 户 has no example in either conditional pool. Counts are reported
for every development target. Missing conditional examples are not a hard
output vocabulary, and coverage alone does not explain failures on common
forms such as 在.

The GPU stage takes 7.815 seconds at 0.311 GiB peak allocation, with no pressure
waiting. The parent contextual body is never loaded. Independent CPU recount
checks every position, motor-error value and all 512 displayed raster tiles.
All eight complete boards were inspected; no pixels were repaired. The
complete-answer results remain 2/32 fixed and 7/32 proportional exact QA,
with no credited complete book continuation. This diagnostic neither qualifies
language nor establishes an efficiency advantage.

The [complete evidence](evidence/glyph_context_motor_capacity_learning_20261111.json),
[registration](evidence/glyph_context_motor_capacity_registration_20261111.json)
and [pixel archive](evidence/glyph_context_motor_capacity_20261111/pixels.npz)
bind the data, scores, source and resources. Boards
[1](evidence/glyph_context_motor_capacity_20261111/board_01.png),
[2](evidence/glyph_context_motor_capacity_20261111/board_02.png),
[3](evidence/glyph_context_motor_capacity_20261111/board_03.png),
[4](evidence/glyph_context_motor_capacity_20261111/board_04.png),
[5](evidence/glyph_context_motor_capacity_20261111/board_05.png),
[6](evidence/glyph_context_motor_capacity_20261111/board_06.png),
[7](evidence/glyph_context_motor_capacity_20261111/board_07.png) and
[8](evidence/glyph_context_motor_capacity_20261111/board_08.png)
preserve every selected position and all three controls. This is a labelled
physical comparison, not a blind language review or independent human study.
