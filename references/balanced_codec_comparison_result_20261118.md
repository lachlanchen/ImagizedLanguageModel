# Equal-form exposure improves rare-form fitting but fails physical qualification

Both new 64-dimensional/64-bit codecs complete all 6,000 updates and fail seven
of ten registered conditions. They start from the original natural-frequency
initial tensors, keep the same architecture, precision, loss and update budget,
and change only codec training allocation. The old controls are not rerun.

| Measurement | Natural continuous | Equal-form continuous | Natural binary | Equal-form binary |
|---|---:|---:|---:|---:|
| Training pooled F1 | .991586440 | .997578104 | .957150751 | .943666840 |
| Training macro-form F1 | .966174790 | .996662318 | .856756069 | .920658411 |
| Validation pooled F1 | .932736083 | .923484240 | .810400756 | .820592454 |
| Validation macro-form F1 | .906819307 | .903684546 | .785655237 | .801056845 |
| Validation exact cells | 68 | 424 | 0 | 0 |
| Validation macro exact rate | .002100840 | .002100840 | 0 | 0 |

Validation has the same 4,907 cells and 952 physical forms in all columns.
Exact continuous cell counts change while its macro exact rate remains two
forms out of 952; repeated occurrences make cell counts and form counts
different quantities. Preserve both. This physical partition has informed
the intervention and is reused development, not blind confirmation.

Equal-form exposure improves rare-form training error and binary validation F1
by .010191697, less than the registered .02 gain. It fails the absolute .98
conditions, .80 macro exactness, retention and common continuous ceiling.
Binary source mean plan-plus-correction work is 43.417590 events, which passes
the unchanged complete-source event condition against 97.302489 direct-tree
events. Both new codecs have zero source false blanks and false ink. Those
three passes do not qualify the writing component or language.

The result supports an effect of training allocation on physical errors, but
not a sufficient repair. Continuous training macro F1 now reaches .996662 while
validation remains .903685; the binary gap persists as well. Do not assume that
more draws, wider codes or a label alias repair solves it. The previously saved
binary codes already distinguish all 9,581 source forms without collisions.
Decoder learnability and representation geometry remain hypotheses, not unique
causes established by this comparison.

## Complete execution and evidence

Each new arm retains 768,000 uniform-form image draws. Weighted complete-source
preflight includes every original training cell and gives each physical form
equal mass. Finite bottleneck/drawing gradients, real updates and exact resets
pass in both modes. Both actual update-three model/optimizer replays are exact.
All original source occurrences, images and hashes remain unchanged.

The [CPU audit](../scripts/export_balanced_codec_comparison.py) recounts all
275,792 initial/final reconstructions, exact XOR recovery and physical event
counts, every one of the 12,000 retained updates and both final tensor hashes.
It checks the preserved natural-frequency control bytes without rerunning
neural fitting or decoding. The [complete result](evidence/balanced_codec_comparison_result_20261118.json)
and fixed [validation](evidence/balanced_codec_comparison_20261118/held_fixed_reconstructions.png)
and [training](evidence/balanced_codec_comparison_20261118/train_fixed_reconstructions.png)
sheets retain actual pixels and all failed conditions. No recognition model,
font correction, repaired transcript or semantic score is used.

The monitored comparison takes 257.158 seconds with .593 GiB peak allocation,
zero pressure wait and 50.607 GiB minimum sampled available host memory. Its
supervisor records 265.267 seconds. The independent audit takes 31.921 seconds
at 1.720 GiB peak RSS. Each codec has 2,030,465 parameters and no contextual
foundation is loaded. These are physical-component costs, not whole-model
language efficiency. All owned model processes exit and foreign GPU1 remains
unchanged.

This fixed equal-form recipe is terminal. Retain its positive macro training
effect together with the failed held-form fidelity. No conditional language fit,
extra epochs, snapshot selection or relaxed threshold follows. The full ILM,
complete-answer, stopping, historical-form and efficiency goals remain active.
