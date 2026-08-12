# Visual Motor Plan V18: Readable Writing Emerges From Continuous Intent

Date: 2026-08-12

## Verdict

V18 is an accepted **topology-first development result** and is not yet a
frozen readable-actuation result.

A `2,358,977`-parameter deterministic visual motor planner learned to turn a
continuous image-derived intent and a separate style image into `32x32` ink.
On a fresh 512-example development audit, correct-state global visual-identity
top-1 is **73.63%**, versus **0.98%** after shuffling only intended states.
Target cosine is `0.8462` versus `0.0716`, and pixel F1 is `0.6577` versus
`0.3129`. Most reviewed simple and medium forms are recognizable. Dense forms
can still merge or lose strokes.

This is a concrete counterexample to the categorical claim that consumer-GPU
image generation cannot learn structured writing without tokens. It is not a
complete language-model result. The intended next state is supplied from a
different-font image rather than predicted autonomously, V16 remains below a
symbolic bigram, and V18's frozen record bank remains sealed.

![V18 development visual motor-plan result](../publication/ilm-image-native/figures/visual_motor_plan_v18_result.png)

## What Changed From V17

V17 asked a stochastic pixel flow to infer glyph topology from a global
`192`-dimensional state while integrating from noise. It learned causal visual
identity but produced mostly pseudo-glyphs. V18 factorizes low-entropy stroke
topology from optional high-entropy surface variation:

```text
different-font intended image -> frozen retina -> continuous intent ----+
                                                                    motor plan -> ink
different-character style image -> continuous style encoder -------------+
```

The motor plan is deterministic. A context MLP modulates a learned `4x4`
spatial seed, conditioned residual blocks expand it to `32x32`, and the final
sigmoid emits continuous ink. No stochastic sampler is needed to decide where
the strokes go. A later renderer may add paper, brush, and font variation after
the topology is stable.

For intent state (z\in\mathbb S^{191}), style image (x_s), and style encoder
(E_s), the planner is

\[
p_\omega=D_\omega\left(z,E_s(x_s)\right)\in[0,1]^{32\times32}.
\]

Training combines stroke-weighted binary cross entropy, soft Dice, pixel L1,
Sobel-edge L1, retinal identity, multi-positive retinal contrast, and a
correct-versus-shuffled state margin. The target pixels supervise the output
but never enter the planner's condition.

## Student Boundary

The learned path receives:

- a continuous `192`-dimensional retinal intent from a writing image;
- a continuous style image; and
- fixed spatial coordinates implicit in convolutional feature maps.

It receives no token IDs, Unicode IDs, strings, OCR, character labels, output
vocabulary, glyph lookup, finite visual codebook, candidate classifier, or
external language model. V17 contributes only the style-encoder warm start.
The V16 retina is frozen. Candidate images and offline duplicate grouping exist
only in evaluation.

This is an isolated actuator experiment. Because the intent is read from a
different-font image of the desired form, the result answers **can a compact
continuous visual state be rendered as readable writing?** It does not answer
**can the causal field decide the next linguistic content?**

## Preregistered Development Selection

Before training, records were assigned by a salted SHA-256 partition:

| Split property | Receipt |
|---|---:|
| Salt | `visual-motor-plan-v18` |
| Training records | `6,573` |
| Development records | `223` |
| Frozen records | `221` |
| Frozen identifier SHA-256 | `1e113b4f...04a79a1` |
| Frozen images instantiated | **no** |

The rule maximized development correct-state pixel F1 while requiring:

1. correct pixel F1 above `0.60`;
2. correct-minus-shuffled pixel F1 above `0.15`;
3. correct identity top-1 above shuffled top-1;
4. correct target cosine above `0.60` and above shuffled cosine; and
5. condition pixel L1 above `0.05`, ruling out direct target-pixel conditioning.

| Step | Correct F1 | Shuffled F1 | Correct top-1 | Shuffled top-1 | Correct cosine | Eligible |
|---:|---:|---:|---:|---:|---:|:---:|
| 200 | 0.5871 | 0.3496 | 62.50% | 0.59% | 0.7226 | no |
| 400 | 0.6308 | 0.3561 | 71.68% | 1.17% | 0.7919 | yes |
| 800 | 0.6663 | 0.3460 | 77.15% | 0.78% | 0.8217 | yes |
| 1,200 | 0.6736 | 0.3278 | 77.93% | 0.98% | 0.8311 | yes |
| **1,400** | **0.6890** | **0.3368** | **78.32%** | **1.17%** | **0.8340** | **selected** |
| 1,600 | 0.6808 | 0.3120 | 77.73% | 1.37% | 0.8407 | yes |

The selected checkpoint SHA-256 is:

```text
85aad4af6f1cb7006d851c087cf559f4ee3ef8cec9f041cdaeb4962778fceafa
```

## Fresh Development Audit

After selection, a separate evaluator rendered `128` fresh development
sequences and selected four positions from each. It generated 512 candidates
under a new seed. The evaluator has no code path for the frozen split.

| Development metric | Correct intent | Shuffled intent | Interpretation |
|---|---:|---:|---|
| Global visual identity top-1 | **73.633%** | 0.977% | strong intent control |
| Target cosine | **0.84618** | 0.07162 | gain `+0.77457` |
| Pixel F1 | **0.65772** | 0.31290 | topology gate passes |
| Soft Dice | **0.56547** | 0.29360 | spatial overlap improves |
| Pixel L1 | **0.09535** | 0.17402 | correct intent halves much of the error |
| Ink fraction | 0.17505 | 0.17505 | occupancy cannot explain the gain |
| Style-copy cosine | 0.06858 | - | style exemplar is not copied |

The correct and shuffled branches share the same style image. Their mean ink
fractions are exactly equal, while identity and topology diverge sharply. The
causal signal therefore cannot be reduced to writing more ink or reproducing
the style exemplar.

## Human Review

Paginated continuous and thresholded contact sheets were inspected at full
resolution. Simple forms such as `王`, `子`, `八`, `元`, `不`, `月`, `之`, and
`而` are clear. Many medium forms such as `作`, `所`, `將`, and `開` remain
recognizable. Dense forms such as `緯` and `劉` can collapse into dark merged
strokes, and some complex forms lose a component.

The preregistration required human review but did not define a blinded rubric
or numeric pass threshold. It would be post-hoc to declare frozen-evaluation
permission from this inspection. The defensible decision is:

- readable writing emergence on development: **accepted**;
- broad-complexity readability: **not accepted**;
- V18 frozen evaluation: **withheld; frozen images remain untouched**.

V19 must define complexity strata and a blinded recognition rule before
training.

## Compute Receipt

| Property | V18 receipt |
|---|---:|
| Trainable motor-plan parameters | `2,358,977` |
| Frozen PVF V16 parameters | `16,471,809` |
| Token/classifier parameters | `0` |
| Training updates | `1,600` |
| Logged training time | `458.34 s` |
| Peak allocated CUDA memory | `0.778 GiB` |
| Typical training throughput | about `380-600` selected images/s |
| Device / precision | one RTX 4090 / BF16 |
| Fresh audit generation | `194.42` images/s |

V18 uses fewer than half the trainable actuator parameters and less than half
the allocated CUDA memory of V17 while improving pixel F1 from V17's frozen
`0.4385` to `0.6577` on fresh V18 development renderings. The splits differ, so
that comparison is architectural evidence, not a matched frozen benchmark.

## What This Breaks

The experiment separates three questions that are often conflated:

1. **Can pixels carry discrete-looking writing structure?** Yes. A small
   continuous decoder learns stroke topology without a token output table.
2. **Can continuous visual intent causally control those pixels?** Yes on
   development. Shuffling intent destroys identity and topology while ink
   occupancy stays fixed.
3. **Can the system autonomously learn and generate general language?** Not yet.
   That requires the causal visual field to predict useful intended states and
   remain readable through repeated write-reread steps.

The result argues against treating language generation as one monolithic image
synthesis problem. Writing has a low-entropy structural layer and a
high-entropy appearance layer:

```text
visual language dynamics -> spatial motor plan -> optional stochastic surface
       what to write          where strokes go        how the page looks
```

This factorization is both more efficient and more testable than asking a large
diffusion model to rediscover linguistic identity, stroke topology, and style
from noise in one network.

## Remaining Bottleneck, V19 Result, And V20

V18 compresses the intended form into one global `192`-dimensional vector. That
state preserves identity well enough for many forms but can discard the exact
location and separation of dense components. V19 subsequently exposed the
retina's continuous `4x4x192` field and added it as a zero-initialized residual
to a clean frozen global planner. It preregistered image-complexity strata, a
numeric blinded-recognition rule, and five interventions on a new salted split.

V19 rejected that repair. On its fresh development audit, correct-field dense
F1 (`0.72776`) was nearly unchanged by shuffling (`0.71898`) or zeroing
(`0.72241`) the field. The fixed causal margins failed, human review was not
authorized, and the frozen bank remained sealed. The full record is in
[`spatial-retinal-motor-plan-v19-result.md`](spatial-retinal-motor-plan-v19-result.md).

V20 must therefore make the local field the primary topology route rather than
an optional residual. The next proof should compare capacity-matched global,
field-primary, fused, shuffled-field, zero-field, and local-occlusion arms under
a new preregistration. Global state may control coarse semantics and style, but
must not be able to redraw all fine topology by itself. Only a writer that
passes automatic causality, blinded readability, and a new frozen evaluation
may be coupled to predicted PVF states.

## Reproduction

The complete V18 training command is:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_visual_motor_plan.py \
  --pvf-checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --warmstart-v17 artifacts/visual_state_actuator_v17_pilot/checkpoint_step_0001600.pt \
  --out artifacts/visual_motor_plan_v18_pilot \
  --maximum-steps 1600 --samples-per-epoch 100000 --development-samples 128 \
  --sequence-length 24 --positions-per-sequence 4 \
  --batch-size 32 --num-workers 8 \
  --style-dim 64 --style-base-channels 32 \
  --plan-base-channels 128 --context-dim 256 --dropout 0.05 \
  --stroke-weight 4.0 --dice-weight 1.0 --pixel-l1-weight 0.50 \
  --edge-weight 0.25 --identity-weight 0.05 --contrastive-weight 0.05 \
  --state-margin-weight 0.10 --state-margin 0.03 \
  --duplicate-similarity 0.90 --logit-scale 12.5 \
  --lr 3e-4 --minimum-lr-ratio 0.10 --warmup-steps 100 \
  --weight-decay 0.03 --gradient-clip 1.0 \
  --log-every 20 --validate-every 200 --validation-batches 4 \
  --save-every 200 --sample-count 8 \
  --precision bf16 --device cuda --seed 20260812
```

The fresh development-only audit is:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_visual_motor_plan_development.py \
  --checkpoint artifacts/visual_motor_plan_v18_pilot/checkpoint_selected_development.pt \
  --out artifacts/visual_motor_plan_v18_step1400_development_audit_v2 \
  --samples 128 --batch-size 32 --num-workers 8 \
  --sample-count 32 --sample-columns 8 --device cuda --precision bf16
```

The evaluator refuses to overwrite a receipt and records that no frozen image
was instantiated. Checkpoints and audit artifacts remain git-ignored; protocol,
code, measured figure, and this receipt are tracked.
