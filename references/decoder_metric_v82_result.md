# Decoder-sensitive loss worsens autonomous Chinese writing

The completed matched experiment fails all four declared repair criteria.
Both arms produce **0/32 pixel-exact autonomous answers on seen training
questions**. Adding the decoder-sensitive term lowers foreground F1 from
85.984% to 70.009%. This recipe is rejected for scale-up. No usable ILM
checkpoint or improvement over a conventional language model is established.

| Final output path | Control exact / 32 | Control F1 (%) | Metric exact / 32 | Metric F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| Autonomous, generated-pixel history | 0 | 85.984 | 0 | 70.009 |
| Correct answer history and known length | 0 | 92.430 | 0 | 75.116 |
| Supplied answer codes through the decoder | 28 | 99.9953 | 28 | 99.9953 |

Each arm starts from the same adapted V81 checkpoint and receives 2,000
retained updates with batch four: 8,000 training-example draws per arm. The
sole treatment difference is decoder-sensitive loss strength, zero versus
0.001. The [original protocol](decoder_metric_v82_protocol.md) and
[owning-history execution revision](decoder_metric_v82r2_protocol.md) retain
the sampler, data, budgets and acceptance criteria. Metric-arm exact answers
must reach 24/32, F1 must reach 0.98, and both must exceed the matched control.
Every criterion fails. Neither arm reaches the absolute training-fit criteria.

The complete actor has 7,583,225,971 parameters, including external pretraining;
53,096,961 parameters are trained locally. Peak allocated memory is
18.893 GiB per arm. Recorded invocation times are 3,894.676 seconds for the
control and 4,053.182 seconds for the metric arm, excluding shared setup,
prefill and initial outputs. A discarded interrupted metric attempt adds
unreported elapsed compute to the total project cost; it is not included in
these retained invocation times. See the [recovery record](decoder_metric_v82_interruption1.md).
These measurements do not establish training efficiency at matched language
quality.

The complete [public export](evidence/decoder_metric_v82_result/README.md)
contains 384 byte-identical before/after PNGs, full per-case results and all
32 cases in four galleries. Review the [first eight](evidence/decoder_metric_v82_result/all_cases_00_07.png)
and [last eight](evidence/decoder_metric_v82_result/all_cases_24_31.png): short
outputs can look readable, but longer outputs lose strokes, substitute content
and terminate incorrectly. Exact pixel equality is stricter than readability;
the visible damage also prevents treating this as merely an overly strict
metric. Foreground F1 measures image agreement, not semantic correctness.

A separate [PNG-only position diagnosis](evidence/decoder_metric_v82_position_diagnosis.json)
authenticates all 388 exported PNGs and reproduces all six full-output
summaries independently, without loading a model or training data:

| Writing path | First-block F1 (%) | Remaining-stream F1 (%) |
| --- | ---: | ---: |
| Control, autonomous | 99.467 | 76.237 |
| Control, correct history | 99.467 | 87.724 |
| Metric, autonomous | 97.018 | 47.493 |
| Metric, correct history | 97.018 | 56.682 |

The first block is 64 image columns, not a word or semantic unit. First-block
means cover all 32 cases; remaining-stream means cover the 28 references with
more than one block, so the columns are different case subsets. Missing ink
counts as error and excess generated ink is retained. Six control answers and
eight metric answers are shorter than their references. Correct history
improves writing but does not repair it, localizing a conditional prediction
failure before accumulated output errors alone can explain the result. This
does not distinguish a unique cause among representation transfer, learning,
the conditional objective and sampling.

The original CPU verifier scores all 64 final free answers, checks retained
assisted/supplied coordinates and replays the 64 initial outputs. A subsequent
[standalone checkpoint replay](evidence/decoder_metric_v82_runtime_replay.json)
reproduces case 00's pixels, stop events,
feedback hashes and reader memory exactly. Its file/socket trace records no
training-data or ancestor-checkpoint access and no internet-family connection.
That one-case replay demonstrates restoration, not general language ability
or a separate implementation of the neural computation.

Reproduce the post-hoc position diagnosis with:

```bash
python scripts/diagnose_writing_failures.py \
  --export references/evidence/decoder_metric_v82_result \
  --output artifacts/writing_failure_diagnosis.json
```

The next candidate must establish legible new-string copying and correct,
prompt-dependent answers on held-out questions through actual image output.
Supplying target codes, observing a lower training loss or restoring a
checkpoint cannot substitute for either requirement.
