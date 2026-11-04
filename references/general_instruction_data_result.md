# Broad instruction source: preparation audit

The pinned [Databricks Dolly-15k release](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
provides an English instruction source under CC-BY-SA-3.0. Revision
`bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` is cached with the author README;
the JSONL SHA-256 is
`2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`.

The completed preparation retains all 15,011 records and all eight categories.
Connected components of identical normalized instructions or nonempty contexts
produce 14,427 groups. Deterministic splits contain 12,070 training, 1,470
development and 1,471 sealed-test records, with zero exact instruction/context
overlap. Near-duplicates and shared source articles are not certified absent.

| Complete flat prompt / answer bound | Train | Development | Sealed test |
| --- | ---: | ---: | ---: |
| 480 / 128 cells | 3,384 | 419 | 390 |
| 480 / 256 cells | 5,104 | 614 | 594 |
| 960 / 512 cells | 8,307 | 1,000 | 988 |
| 2,048 / 1,024 cells | 10,852 | 1,321 | 1,301 |

These are full-source retention counts, without answer-aware cropping or
truncated responses. Original paragraph structure remains alongside derived
flat views. Median flat prompt/answer lengths are 88/186 characters.

The current canonical Noto CJK face lacks characters in 393 training, 55
development and 44 test records. Rendering this source unchanged would create
missing-glyph boxes. A future protocol needs a verified fallback or explicit
coverage filter. A separate read-only check finds **zero** uncovered prompt or
reference characters in either existing CMRC split; V61 data is unchanged.

Files are under `data/raw/databricks_dolly_15k/` and
`data/general_instructions/dolly15k/`, including attribution, hashes, category
counts and font coverage. No model calls, image training or test scores were
produced. Source answers are not independently fact-checked. A later registered
training/evaluation protocol is required; this is not language-quality evidence.
