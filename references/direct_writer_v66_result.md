# V66: broad direct reconstruction does not qualify

The fixed 10,000-update run is complete. A 19,374,096-parameter direct writer
trains with the full inherited contextual body resident, drawing from all
12,976 Chinese/English training examples. Independent replay confirms all
160,000 sampled blocks and 12,972 distinct source examples consumed. The
[registered protocol](direct_writer_learning_v66_protocol.md) and its failed
fidelity decision remain unchanged.

| Fixed reused-development audit, 128 blocks | Aligned | Shuffled |
| --- | ---: | ---: |
| Foreground F1 | 56.563% | 18.520% |
| Exact blocks | 3 | 0 |
| Blank blocks | 2 | 2 |

The conditioning gain is 38.043 percentage points, passing the 20-point
requirement. Aligned F1 misses the original 90% requirement, so the component
fails. These outputs reconstruct supplied answer-image features; they are
not answers generated from questions.

Peak allocated memory is 15.804 GiB. The recorded interval is 1,555.127 s,
including training, checkpointing and evaluation after restoration, excluding
initial loading. All seven early replay checks and the extra optimizer replay
pass. The process and owned tmux session have exited normally.

The [independent receipt](evidence/direct_writer_v66_result.json), produced by
[the committed verifier](../scripts/verify_direct_writer_result_v66.py), binds
the original references, all 256 output scores, frozen sources and journals.
The raw run remains in `artifacts/direct_writer_learning_v66_20261105/`.
Report SHA-256: `dac04c8d98f9bf62cbb4dd60db4b39758a6b17413fcda0d62621dc523663ccb3`.
Final EMA SHA-256: `47cf89ce148a1d7f4d0eea0248dae3cdef4bb51b419883ed772ca85af8063968`.

The separate [V67 audit](codec_output_v67_result.md) qualifies an existing
writing codec as an alternative output space. Neither result establishes
autonomous language understanding or generation.
