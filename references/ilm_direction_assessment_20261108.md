# The writing route is still experimental

The current evidence does not establish a working ILM design. The latest
image writer improves some copies, but cannot yet write requested Chinese
reliably. No natural-question answering has been trained or evaluated in this
writer. Scaling this checkpoint to books would get ahead of the evidence.

## What the actual outputs establish

The [complete 32-update continuation](autonomous_writer_result.md) changes
independent OCR-exact copying as follows:

| Split | Starting checkpoint | Final checkpoint |
| --- | ---: | ---: |
| Seen, Chinese | 1/16 | 3/16 |
| Held out, Chinese | 2/16 | 3/16 |
| Seen, English | 5/16 | 12/16 |
| Held out, English | 7/16 | 10/16 |
| Held out, combined | 9/32 | 13/32 |

OCR is imperfect, including on clean references, so these are OCR observations,
not manual reading accuracy. The complete raw galleries also show obvious
omissions, repeated characters, duplicated lines and copied instructions.
Those visible failures independently reject the existing requirement of
30/32 correct, legible requested strings alone in each split. The English
gain must not obscure the weak Chinese result.

The runtime is an externally pretrained 3.900B-parameter transformer plus
an 84.046M-parameter VAE. Locally training 24.359M adapter parameters does
not make it a 24M foundation model or prove efficient language acquisition.
Fitting this experiment within one GPU is an execution result.

## Why low losses have coexisted with wrong pages

There are measured problems, but no unique causal explanation yet:

- The earlier assisted loss receives a noisy mixture containing the correct
  answer. Actual inference starts from independent noise. The
  [complete reader diagnostic](autonomous_writing_loss_result.md) finds much
  higher loss on autonomous pages. Different checkpoint states and noise
  conditions prevent interpreting this gap as a controlled causal estimate.
- The frozen training reader sometimes transcribes duplicated writing as
  the expected string. Held-out case 026 visibly duplicates its line while
  receiving low target NLL. A reader reward alone can therefore miss physical
  errors that a person sees. It is not the independent evaluation metric.
- In the separate [continuous-code actor](decoder_metric_v82_result.md),
  supplied correct codes reconstruct well, but predicting codes from the
  prompt fails even with correct answer history. Representing a glyph and
  knowing which glyph to write are different learned functions. This does
  not identify one universal bottleneck shared by every architecture.

The latest final-image loss trains through the actual 50-step sampling path
and adds balanced ink/background error. It improves the aggregate copying
observation. That is a reason to investigate further, not proof that the
loss fixes these problems: extra training, corrected input precision and
loss terms changed together. The subsequent additional-training control below
holds native input precision common, but retains different loss weights and
computation. There is no independent-seed replication. Successful
differentiable execution is not language understanding.

## Next decision

The [completed additional-training control](autonomous_writing_control_result.md)
continues the same starting checkpoint for 32 updates with the old objective,
matching native input precision, example draws and optimizer settings. It
gives 5/32 seen and 10/32 held-out OCR-exact copies, versus 15/32 and 13/32 for
the final-image pilot. Both give 3/16 held-out Chinese copies; all net held-out
gains are English. The comparison meets its exploratory rule, specified after
knowing the pilot and before observing the control. This supports further
investigation of the objective bundle, not a causal attribution to one term,
a compute-matched gain or a reliable Chinese writer.

The next [balanced-exposure protocol](balanced_autonomous_writing_protocol.md)
specifies 128 more full-trajectory updates, with four complete passes over
the same 32 training examples. It guarantees 64 Chinese and 64 English draws,
compared with thirteen Chinese draws covering nine unique inputs in the pilot.
Its criterion requires improved held-out Chinese copying and retained English
and seen performance before further scaling. It is in progress, not a result,
and does not isolate additional training from the changed exposure schedule.
A favourable result would still need replication and reliable new writing.
It would not establish semantics. A later
language test must answer two different questions about the same unseen
passage, and change appropriately when the question or passage changes.
Keep legibility and answer correctness separate. Neither attractive figures,
a lower reconstruction loss nor a longer training run can qualify a usable
ILM without those outputs.
