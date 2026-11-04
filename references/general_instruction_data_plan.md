# General instruction data preparation

Prepare the pinned Databricks Dolly-15k public dataset as a future broad
instruction source, independently of the running V61 CMRC experiment.
Revision: `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`.
The author card declares English instructions and CC-BY-SA-3.0:
https://huggingface.co/datasets/databricks/databricks-dolly-15k .

Cache the exact raw JSONL and author README with SHA-256. Preserve all
original instruction, context and response fields. Normalize line endings
and outer whitespace only; derive a separate flat-cell rendering view rather
than erasing original paragraph structure. No generated responses, fixed
answer inventory or model calls enter preparation.

Group records by connected components of identical whitespace-normalized,
case-folded instructions or nonempty contexts. Partition groups
deterministically into train/development/sealed-test using a fixed seed/hash.
This prevents exact shared-context and instruction leakage, but is not an
article-level or semantic-near-duplicate guarantee. Do not claim facts in
human responses are independently verified or that source text was absent
from external pretraining.

Record category counts, lengths, split overlap and full-source retention
under several prompt/answer size bounds. Do not answer-aware crop or truncate
responses, discard non-extractive tasks, or change V61 training data. A later
registered protocol must choose rendering, context bounds, training mixture
and evaluation before using this source. The new test partition must remain
unscored until that final evaluation. Preparation itself is not a trained-model
result. Keep raw/prepared data outside Git with license/attribution metadata.
