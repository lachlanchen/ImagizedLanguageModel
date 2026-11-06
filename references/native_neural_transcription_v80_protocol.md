# Neural reading diagnostic after native training and the fixed audit

This auxiliary protocol is specified after the 20,000-update training images
and failed CPU calibration, before final-model development images or neural
calibration results are available. It changes no model, checkpoint selection,
original image metric or primary gate. Reused development data remain reused.

Use the existing local Qwen3-VL 8B instruct Q4_K_M release, with manifest
SHA-256 `0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319`
and body SHA-256 `1329cd5ab37e5ef6faef37cd8ff8660ffc96ed5b47b34fddf8eb0912e6c82466`.
Verify every manifest asset before loading it. Reuse the shared model store;
do not pull or duplicate weights. This evaluator shares external pretraining
with the actor's reader. Its errors are not statistically independent of that
foundation model, and transcription agreement is not ground-truth reasoning.

The input is writing pixels alone, never a question, paragraph, reference
string, source ID or condition label. Reuse the fixed answer-only instruction
and JSON schema from the earlier blind reader. Set temperature zero, top_p
one, seed 20268040, context 8192, output limit 1024 and thinking disabled.
Each request contains a single anonymous PNG. No prompt/parameter retries,
response repair or substitution are allowed; preserve raw responses and errors.

Reuse all 288 CPU-calibration cases with manifest SHA-256
`5fed1616f03b9ca4fd6791dda00c5bfb08b8ed0a93c944d2c45a574bc996adf7`.
Recover their original 32-pixel-high strips without altering a pixel. Fold
strips longer than 1024 pixels into successive rows: prefer the last entirely
white column in the final half of the row; otherwise cut at 1024. Add four
white pixels between rows. Enlarge fourfold by nearest neighbour and add a
sixteen-pixel white margin. Verify exact reverse reconstruction. This avoids
the oversized-strip error without removing content. The transformation and
all input hashes are frozen before neural recognition.

Calibration requires at least 122/128 exact normalized references separately
for Chinese and English, all sixteen blanks empty, at least fifteen of sixteen
noise controls empty and valid consistent JSON throughout. NFC and whitespace
removal are the only text normalization. Require a Boolean `readable` and a
string `text`; unreadable writing must have empty text. A failed calibration
source receives no actor transcript score. No actor image is recognized if
both sources fail. Keep all calibration failures in their denominator.

After successful source-specific calibration, recognize every corresponding
normal and question-erased output from the original 192-question audit. Verify
raw tensor/PNG hashes and signed physical pixels first. Empty tensors are
known empty writing and need no model call. Freeze all recognition responses
before loading original development reference strings for scoring. Compare
with all original source references; English results are reference agreement,
not a semantic-equivalence judgment. Retain every case. Invalid responses get
agreement intervals [0,1], never an automatically wrong or dropped answer.

Report exact transcript agreement and bounded character accuracy intervals.
For Chinese, also report both-correct pair intervals and normal-minus-erased
character-accuracy intervals. Use the existing article bootstrap, 2,000 draws,
seed 20266379, on lower and upper differences to form a sensitivity envelope.
It does not cover reader errors, training-seed variance or development reuse.
No auxiliary score replaces or changes the original physical-image decision.

CPU-only preparation is allowed during training. GPU startup requires the
actual original pipeline to be complete, 40,000-update checkpoint/report
agreement, independent physical verification and all three standalone CLI
checks. Acquire the project GPU lock, verify GPU0 is free, available RAM at
least 24 GiB and swap use at most 75%, and bind a separate loopback port. Never
call the shared GPU1 service. Start one owned evaluator and terminate its exact
process group and identified descendants on completion or interruption. Record
cleanup and sampled memory, with a 20 GiB GPU-resident ceiling and four-hour
diagnostic ceiling. Incomplete calibration, execution or cleanup produces no
qualified language claim. The deployed ILM never uses this evaluator.
