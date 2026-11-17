# Coordinate improvement does not improve conditional writing

The paired read-only comparison finds **no pooled physical improvement** from
the final profiled reader. All 552 prescribed next-span glyphs per arm are
generated with matched noise in the same frozen learned-context actor.

| Complete training record | Positions | Fixed reader F1 | Learned reader F1 |
|---|---:|---:|---:|
| Book | 512 | .354416895 | .354246782 |
| Short QA | 5 | .331360947 | .333333333 |
| Long QA | 35 | .340766360 | .341267670 |
| Pooled | 552 | **.353468785** | **.353352368** |

Each arm has nine exact source masks, one false blank and four false-ink
outputs at the four true source-space positions. There are no exact QA masks
in either arm. These are canonical physical measurements, not answer accuracy.
The complete sheets retain damaged forms, digits and recognizable individual
characters; no complete language score or repaired transcription is assigned.

Correct reference-image history, span/update counts and next-span lengths are
supplied. Every predicting state precedes its current target. This removes
autonomous grouping, generated feedback and stopping from the test. It does
not demonstrate that either reader independently interprets a new prompt.
The three complete records belong to fitting support, not a held-out sample.

Both installed readers and every other model tensor are unchanged throughout.
The output sample hashes match between arms. The comparison takes 86.565
monitored seconds, including 10.009 seconds of pressure waiting, with 6.055 GiB
peak allocated GPU memory and 52.488 GiB minimum sampled available host RAM.
The outer supervisor records 95.095 seconds, including the monitored interval.
The whole actor has 1,475,346,710 parameters and inherits language pretraining.

The live whole-record frontend differs from the frozen-query cache: feedback
relative RMS is .0530--.0574 for control and .0570--.0622 for treatment. The
export separately measures the raw-query discrepancy. This batch/precision
difference is retained as a limitation; this experiment does not identify its
cause or establish that it explains the writing failure.

The [public evidence](evidence/reader_writing_consequence_result_20261118.json)
binds every raw array, query capture and terminal receipt. An independent CPU
export recomputes all 1,104 tiles' pixel and whitespace metrics and verifies
every displayed pixel against its raw binary array. All six complete sheets
are in the [image directory](evidence/reader_writing_consequence_20261118/).
Sampling-noise agreement is authenticated from the generation receipts; noise
arrays were not separately replayed by this CPU export.

The original .10 reader gate remains failed. There is no new trained actor,
autonomous language result or full-fit admission. This result rejects treating
the measured coordinate reduction as a sufficient repair. The next bounded
candidate tests [conditioning inside the actual image writer](writer_layer_conditioning_direction_20261118.md),
with identity initialization, a matched control and real-image fitting/reset
before considering a larger comparison. No repeated coordinate fit is selected.
