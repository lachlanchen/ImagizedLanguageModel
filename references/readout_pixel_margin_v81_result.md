# Small initial-image errors coexist with incorrect later predictions

The continued frozen-transition control improves its initial answer block
substantially, while complete autonomous answers still fail. A CPU-only audit
of retained correct-history predictions separates the first block from every
later block on the same 32 training questions. No output is repaired or
regenerated, and no adapted-arm output enters this measurement.

| Completed model and block group | Blocks | Incorrect pixels | Mean block foreground F1 | Mean code MSE |
| --- | ---: | ---: | ---: | ---: |
| Native V80, first | 32 | 7,215 | 71.544% | 0.60627 |
| Native V80, later | 75 | 31,650 | 27.326% | 1.56837 |
| Continued frozen control, first | 32 | 182 | 99.330% | 0.03499 |
| Continued frozen control, later | 75 | 21,400 | 55.075% | 0.81290 |

Every target block contains ink. The control's first blocks each have 1–16
incorrect pixels and none is pixel-exact. Only one later block is exact in
each model. These block averages weight blocks equally; they are not the
existing complete-answer score. The continued control's complete free-answer
F1 remains 71.177%, with 0/32 exact answers, and its correct-history answer F1
is 77.358%. All are training-set measurements, not a language qualification.

Decoder logits favor white when positive. A target bit gives the signed margin
(+1 for paper, −1 for ink) times that logit. Of the 182 incorrect first-block
pixels, only 11 have margin in [−0.25,0), while 32 lie below −8. Among the
21,400 later errors, 543 are near zero by the same definition and 10,665 lie
below −8. Thus the incorrect writing cannot be explained entirely by tiny
threshold perturbations. The later blocks include substantial incorrect ink
decisions under correct answer history.

All 214 predicted blocks replay through the frozen decoder on CPU with the
same hard bits as their saved GPU images. Independent NumPy checks rebuild
every target from authenticated bit-packed training files, recompute pixel
counts, signed-margin bins and balanced cross entropy, and verify the source
bindings. This does not independently replay the contextual body or sampler.
The [receipt](evidence/readout_pixel_margins_v81_result.json) retains all block
measurements, source hashes and verification scope; the decoded-logit tensors
remain in the experiment artifacts with their recorded digest.

A decoder-aware training objective is a possible follow-up, not an established
repair. It would need to address actual wrong predictions, preserve coherent
conditional generation and pass the original autonomous-image tests. The
matched adapted arm continues under its unchanged protocol. No semantic
accuracy, successful repair or general ILM capability follows from this audit.
