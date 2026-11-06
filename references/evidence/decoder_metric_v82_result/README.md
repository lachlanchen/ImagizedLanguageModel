# Matched decoder-sensitive writing result

The metric arm fails the declared training-fit comparison. This is a seen-question experiment, not a qualification of general language.

| Final writing path | Control exact / 32 | Control F1 (%) | Metric exact / 32 | Metric F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| Autonomous | 0 | 85.983512 | 0 | 70.009368 |
| Correct history and known length | 0 | 92.430062 | 0 | 75.116411 |
| Supplied-code reconstruction | 28 | 99.995316 | 28 | 99.995316 |

Each arm restores the same V81 adapted checkpoint and receives 2,000 updates, batch four. Only the decoder-sensitive loss strength differs. The original quality criteria are retained:

- `free_exact_at_least_24`: fail.
- `free_f1_at_least_098`: fail.
- `free_exact_exceeds_control`: fail.
- `free_f1_exceeds_control`: fail.

The complete actor retains 7,583,225,971 parameters, including externally pretrained weights; 53,096,961 are trained. Image-agreement scores are not semantic accuracy. Independent CPU scoring and retained-coordinate decoding do not replay the entire contextual body and sampler.

All before/after PNGs are unchanged. The four plates show every case in its original order, including wrong and empty answers:

- [Cases 00–07](all_cases_00_07.png).
- [Cases 08–15](all_cases_08_15.png).
- [Cases 16–23](all_cases_16_23.png).
- [Cases 24–31](all_cases_24_31.png).

[Full provenance, per-case results and CPU verification](result.json). Passing training fit would still require new-prompt/copy and development tests. The general ILM goal remains unachieved.
