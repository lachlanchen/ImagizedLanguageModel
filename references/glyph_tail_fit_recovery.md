# Continue the fixed contextual fit after an execution stop

Recovery retains the original 18,000-step optimizer schedule, source prefix
files, seed, parameter groups, loss, sampler and pixel feedback encoder. It
loads an actual periodic checkpoint, including AdamW moments and CPU/CUDA RNG
states. It does not treat a restarted optimizer as a continuation. Unsaved
updates after that checkpoint cannot be recovered and must be disclosed.

The independent script requires the original fit and coordinator owners to
have exited and uses the same project GPU mutex. It preserves original statuses,
logs and checkpoint. Available host RAM must still exceed 48 GiB at entry and
32 GiB while training; allocated device memory is capped at 12 GiB. Its cumulative
pressure-wait allowance is 1,800 seconds rather than the original 300 seconds.
This execution-resource amendment leaves the neural recipe unchanged. The new
attempt retains its own wall-time receipt; it must not be presented as the
complete cross-attempt time or memory history.

Before continuing the long fit, execute four genuine training updates, save an
actual checkpoint, execute the following four updates, then reload that saved
checkpoint and repeat the latter four. A four-update interval includes three
question sequences and one copy-rehearsal batch. Require bit-identical losses,
sample identities, trainable parameters, optimizer tensors and CPU/CUDA RNG
states. These four replayed optimizer updates are discarded; only one trajectory
contributes to the final 18,000-step checkpoint. This check uses training data
only and is evidence of correct continuation, not answer quality.

The original log reports every 100th update, which coincides with copy rehearsal.
Its displayed pixel losses are therefore not question-answering losses. Recovery
also logs the next question update. Neither trace substitutes for complete
autonomous image-answer assessment. The reserved-question review and matched
readout-only control remain required before either fresh-reference score.

No frozen training source is edited in place. Completed recovery publishes a
hash-bound final checkpoint and receipt at the common fit interface so the
existing qualification and inference commands can consume it. Qualification
must finish and its actual output images must be reviewed before assessment.
