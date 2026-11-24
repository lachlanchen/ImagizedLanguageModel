# Referring to a name outside the answer alphabet

The original sentence writer had no externally passing candidate on 179 out of 1,847 Chinese restaurant cards. Replaying its selected strings exactly showed that only ten candidate pools failed exclusively on names. Another 167 failed exclusively on other fields and two failed both criteria. Correct spelling alone could therefore recover at most ten pools. Both joint failures would also need other repairs. [Error decomposition](evidence/variable_ink_v1/prior_coverage.json)

The new model separates a role's linguistic use from its spelling. Known printed field labels bind original venue and landmark spans. An image writer learns human sentences with two additional actions that address those spans. At emission time the original cells replace the chosen actions. Normalizing name and landmark values to placeholders is established in E2E generation. Here the referent consists of arbitrary physical images. The emitted marks need no transcription or class in the output palette. A Lean proof states the substitution and support properties of this renderer. It requires a correct plan to place each mark in its intended grammatical role. [Earlier E2E system](https://ufal.mff.cuni.cz/~odusek/E2E/final_papers/E2E-TNT_NLG2.pdf) · [Checked source](../formal/span_list/VariableInk.lean)

In 33,512 eligible human training descriptions all 47,558 available role mentions could be aligned at least once. Thirteen records with repeated bound roles were excluded. The image writer and listener have 1,360,121 and 216,621 parameters. Training took 116.1 and 11.8 seconds and consumed 12.92 and 1.39 kJ of on-device energy on one RTX 4090 D. The byte writer took 109.2 seconds and 11.02 kJ. The fitted sentence length distribution and the resulting length-bucket schedule differ from the original writer.

| Chinese input, 1,847 held meanings | Literal fact set | Merged price and rating aliases |
|---|---:|---:|
| Prior optical writer | 657 | 1,606 |
| Frozen optical writer, two familiar surrogate names | 694 | 1,566 |
| Learned roles, unrestricted actions | 670 | 1,665 |
| Learned roles, only available source spans | 670 | 1,668 |
| Matched byte input, only available spans | 675 | 1,688 |

The first role model proposed an absent span on three cards. Restricting those two actions to the visible bindings removed the invalid references. All 1,291 family-policy edits gave the desired direction in both answers; both complete merged fact sets passed on 1,100 pairs. The merged and literal detectors have different orderings across systems. A human reading remains necessary for grammatical and factual adequacy. [Initial results](evidence/variable_ink_v1/results.json) · [Qualified actions](evidence/variable_ink_v1/legal.json)

A separate development transfer test kept 128 real meanings and exchanged each pair of unfamiliar names on the same card. For half the meanings, names were previously absent Latin pairs. For the other half, two distinct historical images were taken from a held dictionary headword. Both source marks were outside the old physical output palette. The prior writer reproduced both names on neither arm. The frozen surrogate model and the learned role model each retained both sets of exact physical names and put the landmark after a recognized proximity phrase on every pair. Requiring all other merged facts gave 58/64 and 49/64 pairs for the frozen model and 59/64 and 56/64 for the learned model. This test verifies a restricted role-direction grammar. It does not confer a reading of the historical glyphs or require the English writer to invent a new shape. [Paired scores](evidence/variable_ink_v1/renaming.json) · [Source PNGs and returned pages](../models/variable_ink/examples)
