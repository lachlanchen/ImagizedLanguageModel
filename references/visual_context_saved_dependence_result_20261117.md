# Visible dependence remains, alongside stereotyped damaged writing

The [saved-output analysis](evidence/visual_context_saved_dependence_20261117.json)
checks all 792 retained streams and 488 prescribed comparisons without a neural
call, OCR or reading repair. The outputs are not literally identical across
prompts. All 32 final normal QA streams are distinct in each arm. Visible
variation therefore rules out describing this run as one exactly constant answer.
It does not establish correct question handling: the frozen complete-language
scores remain zero readable normal QA in both arms.

The learned-context actor nevertheless produces a more recurrent first glyph in
these fixed draws: its final 32 normal QA outputs contain fourteen distinct first
tiles, with one appearing ten times. The frozen control has 32 distinct first
tiles. At initialization both have 29 nonempty normal QA streams and 29 distinct
first tiles. These are pixel identities, not recognized characters or semantic
categories.

| Final comparison | Learned context: matched first-span shapes | Identical first spans | Median first-tile Hamming distance | Frozen control: matched shapes | Identical first spans | Median distance |
|---|---:|---:|---:|---:|---:|---:|
| Two questions about the same passage | 16/16 | 5/16 | 1 of 1,024 pixels | 14/16 | 0/14 | 16 pixels |
| Normal versus question erased | 32/32 | 9/32 | 2 pixels | 28/32 | 0/28 | 12 pixels |
| Normal versus passage erased | 32/32 | 0/32 | 9 pixels | 28/32 | 0/28 | 45 pixels |
| Normal versus erased long-book prefix | 12/12 | 0/12 | 14.5 pixels | 10/12 | 0/10 | 89 pixels |

Distances in this table use only pairs with nonempty outputs and equal first-
span lengths, under the registered common seed and eight flow steps. The
control's excluded cases remain in the complete result; the matched subsets
differ between actors. These medians do not establish a population effect or a
uniform effect on every input. The short-book cohort separately retains its
identical first-step summary; it is not substituted for the long-book cohort.

No compared normal/erased or paired-question case has an exactly equal complete
nonempty stream. Stopping and caps remain separate fields. Small visible changes
may reflect limited conditional information, a damaged common writing pattern,
or other dynamics. Neither equal first pixels in a draw nor different first
pixels proves whether the full conditional distribution understands the prompt.

## Scope and next decision

The source-bound CPU analysis follows its
[prospective descriptive protocol](visual_context_saved_dependence_protocol_20261117.md).
It verifies the original generation manifest, request/report identities, frozen
score cohort identities and all raw-array hashes. It finishes in 0.756 seconds
with 654,295,040 bytes peak process RSS, within its sixty-second/one-GiB limits.
Original scores, frozen readings, checkpoints and images are unchanged.

Together with the completed gradient diagnostic, these data do not justify an
unchanged epoch, a claim of literal prompt independence or an assumed harmful
auxiliary update direction. The remaining problem is useful conditional image
selection and writing. A next intervention should directly train discrimination
between different source-conditioned continuations and preserve a general image
output law. Existing failed contrastive, joint-density and writer trials must
be reviewed before selecting that intervention; repeating their unchanged
recipes is not admitted. Any new conditional objective needs one bounded change,
an unchanged control, complete authentic training records, update/reset support
and fixed complete-language evaluation. No new full fit or working algorithm is
qualified by this saved-output analysis.
