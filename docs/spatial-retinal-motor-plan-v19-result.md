# Spatial Retinal Motor Plan V19: Development Result

Date: 2026-08-12

Status: **automatic development gate failed; frozen evaluation forbidden**

![V19 measured development result](../publication/ilm-image-native/figures/spatial_motor_plan_v19_result.png)

## Result

V19 tested whether a frozen `4x4x192` pre-pooling retinal field could repair
dense writing when added as a zero-initialized residual to a clean global visual
motor planner. The implementation worked, trained on one RTX 4090, and preserved
the image-only student boundary. The causal hypothesis did not pass.

In a fresh deterministic 512-candidate development audit, the correct spatial
field improved dense pixel F1 by only `0.00879` over a batch-shuffled field and
by `0.00535` over a zero field. The prospectively fixed requirements were
`>0.12` and `>0.03`. Correct, shuffled-field, and zero-field pages are visually
almost identical. The adapter learned a small shared polish to the frozen global
planner, not a spatial field whose topology was necessary for the output.

This is a useful negative result. It rules out continuing to attach local visual
features as an optional additive correction while the global branch can already
draw the whole form.

## Fixed Question

V18 compresses the semantic writing image into one global `192`-dimensional
state before drawing. V19 asked:

> Does preserving the retina's continuous spatial field causally improve dense
> writing topology beyond a matched global-only planner?

The protocol, partition, five interventions, complexity strata, and thresholds
were fixed before the non-smoke run in
[`references/spatial_retinal_motor_plan_v19_protocol.md`](../references/spatial_retinal_motor_plan_v19_protocol.md).
No threshold was changed after observing training.

## Clean Baseline And Intervention

A V19-global planner was trained from scratch on the V19 split. It did not load
V17 or V18 learned weights. The selected clean baseline at step 1,600 reached
pixel F1 `0.67541`, identity top-1 `0.77930`, and target cosine `0.83394` on its
development selection run. Its SHA-256 is:

```text
8a3970f8436d7445eecc9b8804685b8665df2252120a4e378a19b0eced2e8eda
```

The V19 intervention froze that planner and the V16 retina, then trained only a
`764,545`-parameter spatial adapter and scalar gate:

\[
(z,F)=R(x_{\mathrm{semantic}}),\qquad
q=q_0(z,s)+\sigma(g)A_\varphi(F,z,s),\qquad
p=D(q,z,s).
\]

The adapter's final projection was initialized exactly to zero, so step zero was
functionally identical to the selected clean global baseline. The target image
entered losses only. It did not enter either conditioning path.

## Fresh Development Audit

The post-training evaluator used seed `22160831`, rendered 128 fresh development
sequences, selected four positions from each, and compared 512 candidates. It
verified checkpoint hashes, the partition receipt, the non-smoke flag, and the
sealed frozen split before generating any output.

| Metric | Correct | Counterfactual | Fixed requirement | Result |
|---|---:|---:|---:|:---:|
| Overall pixel F1 | `0.67104` | - | `>0.68` | fail |
| Dense pixel F1 | `0.72776` | - | `>0.58` | pass |
| Dense F1, shuffled spatial field | `0.72776` | `0.71898` | gain `>0.12` | fail (`+0.00879`) |
| Dense F1, zero spatial field | `0.72776` | `0.72241` | gain `>0.03` | fail (`+0.00535`) |
| Identity top-1 | `0.72656` | `0.00195` both shuffled | `>0.75` and above counterfactual | fail |
| Target cosine | `0.84158` | `0.05787` both shuffled | `>0.84` and above counterfactual | pass |
| Condition pixel L1 | `0.16113` | - | `>0.08` | pass |
| Semantic-target pixel L1 | `0.12513` | - | `>0.05` | pass |
| Frozen images instantiated | `false` | - | `false` | pass |

The identity result confirms that the frozen global path still carries intended
form. It does not rescue the spatial hypothesis: shuffling or removing only the
spatial field barely changes the generated page.

Additional mechanism diagnostics are:

| Diagnostic | Value |
|---|---:|
| Learned spatial gate | `0.13030` |
| Spatial residual RMS | `0.12647` |
| Correct versus shuffled-field pixel L1 | `0.01373` |
| Correct versus zero-field pixel L1 | `0.01172` |
| Simple / medium / dense examples | `180 / 212 / 120` |
| Audit generation throughput | `250.77` examples/s |

## Gate Decision

Four automatic requirements failed: overall pixel F1, dense spatial-shuffle
margin, dense zero-field margin, and identity top-1. Therefore:

- no V19 checkpoint is development-selected;
- the blinded 48-example human review is not authorized;
- the frozen evaluator is not authorized;
- frozen images remain uninstantiated; and
- V19 cannot be described as a dense-topology improvement.

The audited checkpoint SHA-256 is:

```text
82cf4acc392704d565527d38e76766faeb8a1947edd8741a6f5f0582b8357788
```

## Compute Receipt

| Property | Receipt |
|---|---:|
| Trainable V19 parameters | `764,545` |
| Frozen global planner parameters | `2,358,977` |
| Frozen PVF V16 parameters | `16,471,809` |
| Token/classifier parameters | `0` |
| Global-baseline updates / time | `1,600 / 333.48 s` |
| Spatial-intervention updates / time | `1,600 / 329.51 s` |
| Spatial peak allocated CUDA memory | `0.899 GiB` |
| Device / precision | one RTX 4090 / BF16 |

## What The Result Means

The data support three statements:

1. The V19 implementation is a valid, inexpensive, image-only causal test.
2. The existing global state remains the dominant identity and topology route.
3. An optional additive spatial residual is not sufficient to make local
   retinal topology causally important.

The audit does not establish that the `4x4` field lacks useful information.
Because the global planner can already solve much of the task, gradient descent
can reduce ordinary reconstruction loss without using the field strongly. The
experiment therefore rejects this *routing mechanism*, not every spatial model.

## Enhanced Next Goal: Topology-Necessary V20

V20 must make local visual state structurally necessary instead of asking a
small residual to compete with a complete global writer. The bounded design
target is:

\[
p=\sigma\left(U(G(z,s)) + (I-UD)H(F,s)\right),
\]

where `G` can control only a coarse low-frequency plan, `H` receives the spatial
retinal field, `D` is a fixed downsampler, and `I-UD` projects the local route
onto detail that the global route cannot represent. A stricter variant removes
the global spatial seed entirely and lets `z` provide only spatially uniform
channel modulation to the field-driven decoder.

V20 should be preregistered as a matched architecture comparison with:

1. a field-primary topology route and capacity-matched global-only control;
2. correct, shuffled, zero, and locally occluded field interventions;
3. fixed simple/medium/dense gates and the existing blinded rubric;
4. quality, FLOPs, throughput, and peak-memory reporting;
5. no frozen query until every automatic and human gate passes; and
6. no larger language model or page generator until this isolated causal route
   works.

The wider ILM goal remains unchanged: predict useful continuous visual language
state, render it as readable image-native writing, reread the generated pixels,
and eventually answer the Visual Word-Origin Book benchmark independently.

## Reproduction

Train the clean V19-global baseline:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_visual_motor_plan.py \
  --pvf-checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --out artifacts/visual_motor_plan_v19_global_baseline \
  --partition-salt spatial-retinal-motor-plan-v19 \
  --maximum-steps 1600 --samples-per-epoch 100000 \
  --development-samples 128 --sequence-length 24 \
  --positions-per-sequence 4 --batch-size 32 \
  --precision bf16 --device cuda --seed 20260812
```

Train the frozen-base spatial intervention:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_spatial_motor_plan.py \
  --pvf-checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --global-checkpoint artifacts/visual_motor_plan_v19_global_baseline/checkpoint_selected_development.pt \
  --out artifacts/spatial_motor_plan_v19_pilot \
  --maximum-steps 1600 --samples-per-epoch 100000 \
  --development-samples 128 --sequence-length 24 \
  --positions-per-sequence 4 --batch-size 32 \
  --precision bf16 --device cuda --seed 20260812
```

Run the fresh development audit:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_spatial_motor_plan_development.py \
  --checkpoint artifacts/spatial_motor_plan_v19_pilot/checkpoint_latest.pt \
  --out artifacts/spatial_motor_plan_v19_step1600_development_audit \
  --samples 128 --batch-size 32 --num-workers 8 \
  --sample-count 32 --sample-columns 8 \
  --precision bf16 --device cuda
```

The machine-readable result is
`artifacts/spatial_motor_plan_v19_step1600_development_audit/evaluation.json`.
Training and audit artifacts remain git-ignored; this document and the measured
figure preserve the result in the repository.
