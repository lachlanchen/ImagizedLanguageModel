# The fixed isolated-cell reader is unsuitable for training reward

The local Qwen3.5-9B Q8 transcription setup fails a necessary qualification
condition. Among 273 observed distinct nonwhite clean-reference masks, 203
are not correctly read: 202 wrong-character readings and one abstention.
Seventy are correctly read. These failures repeat at 726 reference positions
across the complete 64-case cohort. This is conservative falsification of this
fixed setup, **not a complete calibration or an ILM language result**.

The [registered one-sided analysis](evidence/isolated_glyph_partial_falsification_registration_20261116.json)
was committed and pushed before reference strings or old labels were opened
for scoring. Its [complete evidence](evidence/isolated_glyph_partial_falsification_20261116.json)
retains all 9 known-correct, 9 readable-wrong and 46 uncertain/uncredited cases.
Missing readings remain unknown. There is no selected-subset AUC, imputation,
reference-assisted transcription or repair of a frozen reading. No exact
physical-mask/canonical-character alias occurs in this reference set; this
does not establish identifiability across the full training corpus.

## Execution and scope

The first attempt stops after one request because the shared server is pinned
to GPU1. A separately registered temporary server pins the recovery to the
physical GPU0 UUID and reuses the existing executable and weights. It saves
402/526 responses in 260.087 seconds before the unchanged foreign-device
occupancy guard observes a GPU1 increase and aborts it. This increase does not
establish that the GPU0-pinned teacher uses GPU1. The wrapper exits its owned
server, closes its port and returns GPU0 occupancy to 16 MiB. No foreign service
is stopped and no third teacher run is launched.

There are 124 unexecuted queries, including 102 of the 375 distinct nonwhite
reference masks, and 448 unknown reference occurrences. Completing these
readings cannot repair the observed violations of the prospective requirement
that every clean reference be correctly read. An incomplete run can therefore
reject the judge without being promoted to a completed calibration.

An independent recount reconstructs all 402 requests and matches every saved
payload hash. All responses report 179 evaluated prompt positions. This checks
client pairing, not internal server image consumption. The cell transform
preserves original pixels by nearest-neighbor enlargement, without substituting
a font or corrected glyph. The following unchanged reference examples were
inspected after unmasking. They are the first two distinct mask/expected-character
failures in case order, **not student outputs**.

| Clean reference | Reader output | Original reference pixels |
| --- | --- | --- |
| 实 | A | ![Clean reference 实](evidence/isolated_glyph_reader_samples/reference_01.png) |
| 管 | π | ![Clean reference 管](evidence/isolated_glyph_reader_samples/reference_02.png) |

The [sample manifest](evidence/isolated_glyph_reader_samples/manifest.json)
binds these exact image bytes. The reader declares 401/402 responses accepted;
structured syntax and claimed certainty do not establish accurate reading.
This experiment does not isolate quantization, preprocessing, runtime, prompt
constraints or absent neighboring context. It does not show that the
unquantized model or vision-language models generally cannot read Chinese.

## Costs and decision

There are 403 completed teacher requests across both attempts and zero student
optimizer updates. The first attempt takes 12.559 seconds. Recovery responses
report 71,958 prompt positions, 5,930 generated tokens and 222.060 seconds of
summed service duration. The recovery supervisor including startup/cleanup takes
266.926 seconds. Maximum sampled GPU0 occupancy is 9,876 MiB (9.645 GiB), measured
around requests, not as an allocator peak. The aborted client's minimum host
RAM and final peak RSS were not retained. The later CPU analyzer's 0.609 GiB RSS
must not substitute for teacher resource use. CPU analysis takes 3.394 seconds
and makes no new model call.

Reject this reader for optimization. Preserve both interrupted attempts, the
partial freeze, missing observations, original reviews and source bindings.
No prompt tuning, automatic continuation, student fit or deployment change
follows. This work has not improved the ILM language score. The last complete
language result remains 7/32 strict proportional answers, 0/16 both-correct
question pairs and no credited coherent long-book continuation. The next
[supervision investigation](training_glyph_supervision_direction_20261116.md)
begins with source and image-path verification.
