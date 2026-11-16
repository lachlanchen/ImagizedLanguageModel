# Reserve unscored Chinese reading source groups

The completed language pilot reuses its development cohort and fails. Even a
future pilot pass would require confirmation on new source groups. This CPU
reservation prepares that later decision before inspecting new model outputs.
It does not change the live white-target run, the original pilot or its gates.

Use the official DRCD test source at commit
`b944790de5af02c5fbb7cd9cb1473d27d169eebf`, authenticating exact byte counts and
Git blob hashes for the data and README. The dataset supplies human-annotated
traditional-Chinese reading questions. Its Wikipedia-derived content is
CC-BY-SA3.0; preserve the original notice. Inherited foundation-model exposure
is unknown. A repository/cache search found no existing DRCD use in the inspected
project files; this is not a complete historic or external-exposure audit.
[Official dataset and notice](https://github.com/DRCKnowledgeTeam/DRCD),
[primary paper](https://arxiv.org/abs/1806.00920).

Bind the implementation, every screened source file and all installed OpenCC
package resources before execution. For exclusion screening only, use Unicode
NFKC, existing OpenCCt2s and whitespace removal. Keep all reserved source writing
and human references unchanged. Screen against both complete local CMRC splits,
all prepared public-domain Wikisource records and the entire newly prepared
Chinese preference source. Also exclude the README example article, whose
content has already been viewed. Do not render, display, transcribe, score or
ask a teacher to answer the reserved questions.

Exclude an entire DRCD article if any of these occurs: its normalized title is
known/viewed; a paragraph equals a normalized known document; it contains an
indexed exact64-character source window; or a question exactly matches a
normalized known question. Index known source windows every32characters and
query every candidate offset. This is a conservative exact-overlap screen,
not a proof of semantic or paraphrase decontamination. Unmodified source files,
normalizer hashes and all quarantine reasons remain available for audit.

Require every annotation of an admitted question to be a nonempty exact source
span at its supplied offset. Quarantine invalid questions without repairing
them. Each reserved paragraph needs at least two distinct questions with
disjoint normalized answer sets. Select its fixed question pair by a declared
hash order, retaining all valid questions. Reserve every surviving eligible
paragraph, with no count limit or answer-length filter. Preserve the full raw
test source and all excluded identifiers/reasons; do not select examples after
seeing generation quality.

The raw cache belongs under `data/evaluation_only/drcd_confirmation_v1/`; the
pool and quarantine manifests belong in a new ignored evidence directory.
Their role is evaluation only. Do not use this pool for student training,
reward calibration, preference labels, hyperparameter choice or source repair.
The original fixed comparison remains separate. Reserve initial-checkpoint and
named local-Qwen comparisons for a later explicit execution contract rather than
running them now.

Use one low-priority CPU process with180seconds,1GiB peak RSS,48GiB available
host RAM at entry and24GiB at runtime. Use no neural imports, GPU, image renderer
or extra model job. Preserve any failed or partial download; do not overwrite
existing cache or reservation evidence. Download only the pinned public files.

Success here means a nonempty frozen evaluation-only pool and complete source,
annotation, overlap and quarantine accounting. It is not a language result or
a complete confirmation evaluation. Any missing source, invalid resource bound
or empty pool stops preparation without relaxing the policy. A later evaluation
must first register image preparation, unchanged review/scoring rules, limits,+cold execution, comparators and its decision. Require the current actor's actual
physical qualification and original autonomous pilot before invoking it on
this confirmation source. Book, historical-form, independence and matched-cost
requirements remain separate and unproved.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/reserve_drcd_confirmation.py --fetch \
  --output artifacts/drcd_confirmation_reservation_v1
```
