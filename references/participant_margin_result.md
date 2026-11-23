# Which person did the pictured question ask about?

The passage says **新曲《松窗》由周至作词，苏恬作曲。** Asked for the composer, the previous system credited the lyricist. A learned span reader now emits **苏恬是《松窗》的作曲人。** Both word orders are correct across all four person–role combinations on that song. The song was already known to expose the bug. The independent evidence comes from 128 separate annotated passages and 48 additional constructed credits. [Actual output](evidence/participant_margin_20261123/song_matched.png) · [Registered comparison](participant_margin_protocol.md)

| Every generated answer correct | Inputs per source | Previous | Randomized paired labels | Matched |
|---|---:|---:|---:|---:|
| New DuIE sources, 128 passages | 4 questions × 2 word orders | 29 | 11 | **70** |
| New credits, 48 passages, both clause orders | 8 questions × 2 word orders | 13 | 2 | **36** |
| Human-written questions, 48 inputs | 2 word orders | **34** | 30 | 30 |
| Earlier programme questions, 32 inputs | 2 word orders | **18** | 13 | 17 |
| Earlier DuIE passages, 1,200 inputs | 2 word orders | 892 | 856 | **924** |

The new DuIE passages begin with a recorded passage and two annotated roles for the same subject. Substituting the names changes the factual assignments. Two previously specified questions query the original and edited versions. The additional 48 credits use separate subjects and generated clauses that can also be reversed. Each reported success requires every answer to match its reference pixels and stop. Across the 128 source pairs the updated model gained 44 and lost three complete passages. On the 48 credits it gained 23 and lost zero. Prior text-model pretraining exposure is unknown. The 48 original human questions have previously been used to investigate failures. [All counts and output hashes](evidence/participant_margin_20261123/results.json) · [Paired changes](evidence/participant_margin_20261123/paired.json)

## What the objective guarantees

Let the reader assign a score to every contiguous span of one to 48 glyph cells. Score a span by adding its two endpoint scores. If the score of the true span is *t* and that of a rival is *r*, a tie or incorrect preference incurs at least one unit of loss under max(0, 1 + r − t). The sum of these penalties bounds the number of such pairwise errors. For finite binary scores, scale both by a common positive denominator to obtain the exact rational-score version. Lean checks this bound, identifies the selected span when it strictly beats every competitor, and proves the corresponding result on an edited page when its candidate spans are covered and their score differences are preserved. Coverage and preservation are assumptions about the reader. They require empirical testing. [Lean source](../formal/participant_margin/ParticipantMargin.lean) · [Compiler axiom report](evidence/participant_margin_20261123/lean.txt)

Beating a single rival cannot guarantee a correct name. Lean checks a three-span example in which the pairwise penalty is zero and a different span wins. Training therefore combined the paired penalty with a probability distribution over **all** admissible spans. In the observed 512 new source questions, the correct name beat the annotated rival 472 times, but was selected exactly only 430 times. The other 42 illustrate the gap between pairwise discrimination and reading the whole name. [Score-gap measurements](evidence/participant_margin_20261123/transfer.json)

## What was trained

A 122,882-parameter head combines a frozen contextual glyph representation, five-cell neighbourhoods and a learned embedding of the requested role. It adds a correction to the previous start and end scores. Inference receives passage and question pixels. It chooses among all page spans and supplies the chosen pixels to the existing autoregressive writer. The candidate names used for supervision are unavailable to the inference routine. The original subject and role heads and both pretrained backbones stay frozen. The printed output can copy source ink or select a glyph from the writer's output memory.

The fit used 9,600 original training questions and 9,600 paired questions, derived from 1,200 actual two-role DuIE passages. Both arms received the same five passes and 2,000 updates. The matched fit took 20.8 seconds on one RTX 4090 D. Contextualizing the 9,600 new image pairs took 13.0 seconds. These figures exclude prior language pretraining, preparation of the original reader, and the writer's previous training. [Corpus and split digests](evidence/participant_margin_20261123/corpus.json) · [Fit](evidence/participant_margin_20261123/matched_fit.json) · [Control](evidence/participant_margin_20261123/permuted_fit.json)

A secondary fit removed the rival penalty and kept the same correct labels, initialization, batches and update count. This likelihood-only scorer passed all generated answers on 79/128 original passages and 37/48 clause-reordered credits, compared with 70 and 36 with the penalty. It wrote both word orders on 28/48 human questions and 926/1,200 earlier DuIE passages. This comparison isolates the penalty within one seed. Likelihood alone gave the best complete exchange scores in this comparison and lost two further complete human replies relative to training with the penalty. Its [checkpoint](../models/participant_margin/likelihood.safetensors) is also available. The larger difference from randomized labels measures the combined effect of correct supervision on the two losses. [Likelihood-only fit and full outputs](evidence/participant_margin_20261123/ablation.json)

## Where this version still fails

On 48 human questions, the original pattern writer went from 36 complete answers to 32. The generative writer went from 34 correct in both orders to 30. The first lost object-first answer writes 清吴沃尧 where the reference says 吴沃尧. It identifies the right author but includes the preceding dynasty marker. The exact score penalizes that extra glyph. On 256 broadly sampled human questions direct span accuracy also fell, from 147 to 121. The likelihood-only scorer gave 119/256 on the same direct questions. On 32 historical-name substitutions, complete statements rose from 13 to 14 with the margin and fell to 12 with likelihood alone. [Likelihood-only transfer](evidence/participant_margin_20261123/likelihood_transfer.json) These are physical substitution tests, without a claimed historical reading of the invented names. The candidate failed the human-question criterion and is available as an optional scorer. The earlier default remains installed. [Actual longer answer](evidence/participant_margin_20261123/human_matched.png) · [Broader transfer](evidence/participant_margin_20261123/transfer.json)

## Reproduce or run locally

The repository includes the fitted participant checkpoint. The local distribution in `artifacts/participant_margin_v1/participant-scorers.zip` installs on top of the earlier full model and pictured question update. Pass `--participant-head model/participant.safetensors` in the extracted installation to enable it. The release check runs the earlier and updated routes on CPU and compares their answer images with the GPU outputs. [Inference example](../models/participant_margin/README.md) · [Local release hashes](evidence/participant_margin_20261123/release_both.json)

With the original visual fact model, training cache and DuIE corpus prepared, run the following on a fresh experiment directory.

```sh
lean formal/participant_margin/ParticipantMargin.lean
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
python scripts/prepare_participant_margin.py
python scripts/train_participant_margin.py
python scripts/evaluate_participant_margin.py
python scripts/probe_participant_margin.py
python scripts/train_participant_margin.py --likelihood-only
python scripts/evaluate_participant_margin.py --likelihood-only
python scripts/probe_participant_margin.py --likelihood-only
python scripts/report_participant_margin.py
```
