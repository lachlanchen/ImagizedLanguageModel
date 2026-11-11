# Known first forms reconstruct well, but conditional writing remains unreliable

All 32 canonical first-answer forms are present in the original glyph training
primitives. Supplying those images to the frozen encoder and decoder gives
31/32 pixel-exact reconstructions, with mean foreground F1 0.999934. Missing
first-form coverage and poor direct reconstruction therefore do not explain
the broad difference between the native reader and the image-output model.
This is an answer-supplied physical control, not language generation.

The [registered CPU protocol](glyph_tail_first_writing_protocol.md) preserves
all 192 original QA streams: 32 questions, two input presentations, normal
inputs and two erased controls. All 192 first contextual states match the
completed native diagnosis. The same authenticated rasterizer reproduces
all 4,096 training primitive characters and all 745 canonical target tiles
exactly before neural reconstruction. The inventory contains 4,087 distinct
images. No original GPU emission is regenerated or replaced.

| Original first image against canonical first tile | Pixel-exact /32 | Mean foreground F1 | Pooled foreground F1 |
|---|---:|---:|---:|
| Fixed cells, normal | 0 | 0.492173 | 0.503662 |
| Fixed cells, question erased | 0 | 0.440010 | 0.466943 |
| Fixed cells, passage erased | 0 | 0.471837 | 0.499471 |
| Proportional, normal | 3 | 0.632918 | 0.650936 |
| Proportional, question erased | 0 | 0.429196 | 0.445758 |
| Proportional, passage erased | 0 | 0.465444 | 0.488394 |

These are pixel comparisons, **not first-character or semantic accuracy**.
The full [first comparison board](evidence/glyph_tail_first_writing_20261111/first_glyphs_01.png)
and [second board](evidence/glyph_tail_first_writing_20261111/first_glyphs_02.png)
show every normal case, its canonical tile, supplied reconstruction and both
original first outputs. Some outputs have damaged or superposed strokes.
Others begin a valid alternative wording: for example, the population answer
starts with the number rather than the reference's opening noun. Even the two
previously exact complete fixed-cell answers have nonidentical first pixels.
No semantic score is inferred from these differences and no prior answer
reading or strict matching rule changes. Both labelled boards were viewed
with prior cohort familiarity; there is no independent human adjudication.

| Supplied-image reconstruction | Positions | Pixel-exact | Mean foreground F1 |
|---|---:|---:|---:|
| First canonical forms | 32 | 31 | 0.999934 |
| All canonical answer positions | 745 | 726 | 0.999220 |
| Positions covered by training primitives | 743 | 726 | 0.999933 |
| Positions absent from training primitives | 2 | 0 | 0.734417 |

The two uncovered later characters are 哉 and 倦, one occurrence each. Thirty
of 32 complete answers have full primitive coverage. Neither missing form
occupies a first position. Membership measures training coverage; the
continuous decoder has no hard output inventory, so absence does not prove
that a shape cannot be generated. No evaluator inventory enters inference.

Among the seven proportional cases with exact native answers and nonexact
glyph answers, one first image is pixel-exact and the other six have foreground
F1 between 0.433898 and 0.906250. Three of the seven streams reach the
128-glyph cap. The exact first image belongs to the answer whose omitted
parenthetical alias already explains its strict mismatch. Consequently,
neither first-image damage nor later feedback explains every failed answer.
No stream stops before its first image in any of the 192 cases.

An [independent CPU recount](../scripts/summarize_glyph_tail_first_writing.py)
verifies every integer TP/FP/FN counter, subgroup, coverage flag and denominator
from the saved arrays. It also checks all 128 displayed tiles against exact
two-times nearest-neighbour enlargements of those arrays. The
[complete evidence](evidence/glyph_tail_first_writing_learning_20261111.json)
includes every case and every target position; the
[binary pixel archive](evidence/glyph_tail_first_writing_20261111/pixels.npz)
supports recounting without model weights. Source identities and the
[registration](evidence/glyph_tail_first_writing_registration_20261111.json)
bind the unchanged actor checkpoint
`5c434a9a9b0c6ddc0c3fe4005265e4f8ca5b7d0add9a74a80ed371806fa71085`.

Preparation takes 6.539 seconds without a neural call. Frozen reconstruction
takes 1.502 seconds with two CPU threads and 1.078 GiB peak process RSS.
CUDA remains uninitialized. There are no contextual-model calls, optimizer
updates or new autonomous ILM images. All original streams remain available.

The result locates a conditional-writing problem before feedback in some
cases while preserving known reading, wording, stopping and later-history
limitations. It does not select a successful algorithm. Earlier dense and
spatial semantic-flow tests and the balanced deterministic writer also failed
their relevant generalization gates. The next
[bounded conditional-writer decision](glyph_tail_context_writer_decision.md)
uses those failures rather than repeating them. The independent actor still
scores 6/32 exact proportional answers and 0/16 both-correct pairs; the usable
visual-language goal remains unachieved.
