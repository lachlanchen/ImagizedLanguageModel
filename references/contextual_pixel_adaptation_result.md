# Contextual adaptation does not meet the reader requirements

The completed matched comparison gives **26/64 exact unseen Chinese answers
with contextual adaptation versus 23/64 with the frozen contextual body**.
Both arms answer both questions correctly in only **3/32 paragraph pairs**
and retain **32/32 exact new copies**. Neither qualifies the reader, and
adaptation fails the registered improvement rule. These are source-image
answers, not free image-language generation.

The [verified result](evidence/contextual_pixel_adaptation_result/result.json)
and [complete paired gallery](evidence/contextual_pixel_adaptation_result/README.md)
retain every answer, reference and input. Exact scoring removes only exterior
white columns. It does not forgive extra punctuation, missing marks or
truncated characters. No separate semantic or readability score is claimed.

| Fixed cohort | Frozen control | Contextual adaptation |
| --- | ---: | ---: |
| New Chinese/English copies | 32/32 | 32/32 |
| Previously seen Chinese questions | 28/32 | 27/32 |
| Unseen Chinese questions | 23/64 | 26/64 |
| Both questions correct about the same passage | 3/32 | 3/32 |
| Question ink erased | 1/64 | 0/64 |
| Passage ink erased | 0/64 | 0/64 |

On the same 64 unseen questions, 20 answers are correct in both arms, six
only after adaptation, three only in control, and 35 in neither. Adaptation
adds three exact answers, no complete paragraph pairs and no copies. The
prospective improvement rule requires at least eight additional exact answers,
four additional complete pairs and a copy loss no greater than two. It fails.
These are decision floors, not statistical significance claims.

Both arms pass the copy floor and the 25-percentage-point question-erasure
gain floor: gains are 34.375 and 40.625 points. Both fail the 48/64 natural-answer
and 20/32 paired-answer floors. Normal unordered-column F1 is 61.622% and
67.444%; this order-insensitive image diagnostic is not a language score.

## Matched learning and verification

The [prospective protocol](contextual_pixel_adaptation_protocol.md) and executed
comparison were committed at `01aff95`. Both arms initialize from the same
completed head and zero-output contextual adapters. They use identical four-pass
schedules over **3,193 admitted Chinese QA images and 64 training copies**:
**13,028 draws and 3,257 updates per arm**. Eighteen QA records are excluded
during preparation before model calls. Development outcomes never select a
checkpoint or change the recipe.

Control optimizes **2,378,370 head parameters** from authenticated frozen
features. Adaptation additionally optimizes **6,193,152** rank-eight query,
value and output attention parameters across 36 contextual layers, for
**8,571,522 optimized parameters**. Each complete runtime contains
**7,531,277,171 parameters** and retains external Qwen3-VL pretraining. The
registered learning rates, joint clipping and final-checkpoint rule remain
fixed. This tests the complete regimen, not a unique internal causal mechanism.

The independent audit authenticates every schedule row, all four occurrences
of each training ID, both final checkpoints and finite Adam moments at step
3,257. Fifteen head tensors change in each arm. No contextual tensor changes
in control; all 216 contextual adapter tensors change in adaptation. There
are **zero additional replayed optimizer updates**.

All **512 prescribed arm/input calls** use the full native image path.
Independent NumPy/PIL scoring reconstructs every unmodified output from its
declared original source columns and reproduces all scores, pairs and gates.
The export authenticates **927 retained files**, in addition to its result
and explanatory gallery/cost notes. Model weights and feature payloads are
not copied into public evidence.

Four additional input-only PNG calls from each final checkpoint reproduce
answer pixels and selected events exactly. These eight standalone calls open
no training manifest, feature cache or original head checkpoint. They still
require the shared pretrained source archive and learned checkpoint. Their
fixed inputs are the first two normal development questions and first
Chinese/English copy requests. They are replay checks, not a new test.

For the first fixed question, both arms output common names with an extra
trailing comma; its exact score is false. Their second output matches the
ocean-region reference. These legible cases do not represent the whole cohort;
the gallery retains every wrong answer and boundary error.

The [data/geometry audit](contextual_pixel_data_audit.md) remains material.
Question erasure preserves reference ink but shifts other ink on 11/64 inputs.
Passage erasure removes every admissible correct source span, so this motor's
support forces zero exact answers. That zero cannot isolate comprehension.
Public development is repeatedly reused and is not a sealed test.

## Costs and interrupted assessment

Both arms run sequentially on one RTX 4090 D. Original source setup records
93.118 seconds. Feature preparation takes 3,387.907 seconds for 2,937 new
encodings, with 320 authenticated prior payloads reused. Optimization times
exclude that preparation, additional reloads and external pretraining.

| Optimization | Elapsed seconds | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: |
| Frozen control | 126.665 | 15.265 | 15.504 |
| Contextual adaptation | 30,856.570 | 18.075 | 22.082 |

Control memory includes the resident full runtime; its cached-head time is
not the cost of a small complete model. Four standalone calls take 77.695
seconds setup plus 5.022 seconds inference for control, and 76.854 plus 5.056
seconds for adaptation. Both record 17.948 GiB peak allocated memory. This
small replay cohort is not a matched-quality throughput benchmark.

Assessment is interrupted four times and recovered across five controller
invocations. The original 218 control calls, a further 38 control/72 adapted
calls, and the final 184 adapted calls account for all 512 prescribed calls.
The existing 328 indexed outputs survive the final recovery unchanged.
Optimization never repeats. A persistent project runtime completes assessment
and the standalone loaders, then is removed. The termination sender remains
unidentified; the runtime change does not prove the original stop cause.

Complete assessment and additional reload durations were not measured.
The retained 45.634-second control observation covers only 38 calls; the
223.855-second adapted observation covers only the final 184. A completed-control
resume records almost no current-invocation work and must not be presented
as the time for its 256 calls. The [execution notes](evidence/contextual_pixel_adaptation_result/execution_notes.json)
preserve these boundaries. No total efficiency advantage is established.

This regimen does not qualify for scale-up under its fixed requirements.
It accepts prepared writing strips and can only select source crops, so it
cannot compose an absent answer. Historical writing, handwriting, general
instruction following, a lightweight foundation trained from scratch and
parity with a conventional language model remain unproved. Reliable free
image-language generation is still an unmet requirement.

CMRC2018: Cui et al. (2019), CC-BY-SA-4.0. Qwen3-VL: Alibaba, Apache-2.0.
The retained copying requests are synthetic.
