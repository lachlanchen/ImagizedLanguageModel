# Trainable native readout does not repair autonomous writing

The matched repair fails. On the same 32 training questions, the adapted
model produces no pixel-exact complete answers and reaches 72.748% foreground
F1. The continued frozen-transition control also produces no exact answers,
at 71.177% F1. This comparison does not qualify a useful ILM or establish
semantic accuracy. Its small image-agreement increase is not a reason to
scale the recipe.

| Final-checkpoint writing path | Control exact / 32 | Control foreground F1 | Adapted exact / 32 | Adapted foreground F1 |
| --- | ---: | ---: | ---: | ---: |
| Autonomous, learned stopping, actual pixel feedback | 0 | 71.177% | 0 | 72.748% |
| Correct preceding image codes and known length | 0 | 77.372% | 0 | 79.629% |
| Supplied target codes through the frozen decoder | 28 | 99.9953% | 28 | 99.9953% |

All autonomous streams stop. First-block F1 is 99.330% in the control and
98.174% in the adapted arm. Later writing still deteriorates, including under
correct answer history. Generated-history error is therefore not the sole
cause. The fixed decoder's supplied-code fidelity shows that it can represent
these targets; it does not show that the contextual predictor has learned
which codes to emit. The comparison does not uniquely identify whether the
remaining limitation is representation, optimization or the writing objective.

Both arms start from the same completed native checkpoint and receive 2,000
updates, batch four, the same case/noise draws and 250 exposures per question.
The adapted arm combines a continuous initial read query with rank-8 q/v/o
suffix adapters across the inherited 36-layer body. The composite intervention
has 53,096,961 trainable parameters and 7,583,225,971 complete parameters;
the control has 46,899,713 and 7,583,221,875. External foundation pretraining
and earlier codec learning remain outside the local repair cost. This is
neither training a lightweight language model from scratch nor evidence of
an efficiency advantage over a conventional language model.

The original training execution finishes both arms but stops its evaluation
at adapted case 00. Autonomous and full-history first-block pixels agree;
an over-strict check rejects a stop-probability difference of 3.908e-14.
That execution and its partial outputs remain unchanged. A separately
[specified evaluation](adaptive_readout_v81_evaluation_protocol.md) restores
the two final checkpoints without optimization and uses the same incremental
start/cache path for autonomous and correct-history controls. All 32 prior
control autonomous outputs and the one retained adapted autonomous output
replay exactly. The original failed execution is not relabelled as passed.

The retained full-history diagnostic gives 77.358% control F1 and 79.632%
adapted F1. It differs from incremental control in 50 and 42 physical pixels
across the 107 blocks per arm. Its hidden-state comparisons satisfy the
existing V81r2 normwise criterion; the original absolute 1e-4 comparison fails
on two control cases and one adapted case. Numerical discrepancies remain
explicit and do not explain the large quality shortfall.

CPU verification independently scores all 64 autonomous answers, re-decodes
128 retained control coordinate streams and 64 supplied-code streams, and
checks owning prompts, targets, raw pixels, PNGs and checkpoint/source
bindings. It does not independently replay the entire contextual body and
sampler. Evaluation takes 221.75 seconds, including model setup and prompt
prefill, and peaks at 17.675 GiB allocated VRAM on GPU0. These are evaluation
costs, not training costs.

Three of four original repair criteria fail: at least 24 exact answers,
free F1 at least 0.98, and an exact-count improvement over control. Only the
foreground-F1 improvement criterion passes. The cohort is seen training data;
no new-prompt, general-language or historical-script qualification follows.
The general ILM objective remains unachieved.

The [complete machine-readable receipt](evidence/adaptive_readout_v81_result/result.json)
contains full case scores, original failure, source identities and verification.
All 320 PNGs are byte-identical to their saved sources. The four plates show
every case in its original order, at one image pixel per display pixel:
[00–07](evidence/adaptive_readout_v81_result/all_cases_00_07.png),
[08–15](evidence/adaptive_readout_v81_result/all_cases_08_15.png),
[16–23](evidence/adaptive_readout_v81_result/all_cases_16_23.png),
[24–31](evidence/adaptive_readout_v81_result/all_cases_24_31.png).

Runtime artifacts are in `artifacts/adaptive_readout_v81eval_20261107/`;
the preserved failed training execution is in
`artifacts/adaptive_readout_v81r2_20261106/`. The original native full-data
result remains [separate](native_chinese_v80_result.md).
