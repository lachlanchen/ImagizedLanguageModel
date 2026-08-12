# Predictive Visual Field V16: Residual Causal Visual Memory

Date: 2026-08-12

## Verdict

V16 is an accepted **visual-state replication with a small directional
improvement**, not an accepted language system. It adds a six-million-parameter
multiscale causal visual memory to the V15 recurrent base and remains entirely
inside the image-derived continuous state space.

On the unchanged frozen 512-form benchmark, the selected V16 proposal obtains
**6.264%** top-1 (`112/1,788`) versus **3.971%** last-image-only, **1.734%**
unigram, **0.224%** random dynamics, **0.195%** chance, and **13.143%** symbolic
bigram. V15 obtained **5.872%** (`105/1,788`) on the same frozen contexts. The
seven-example increase is directional evidence, not a statistically established
architecture win.

V16 proves that local visual fields and global causal attention can be added to
the image-only student for `1.479 GiB` peak allocated CUDA memory on one RTX
4090. It does not prove readable image generation, broad language
understanding, etymology question answering, superiority to token models, or
that the memory module caused the small score change.

## Student Boundary

The learned path is:

```text
32x32 writing images
  -> convolutional retina
  -> continuous retinal states
  -> GRU base + residual multiscale causal visual memory
  -> continuous visual proposal + hyperspherical state flow
```

The student receives no strings after rasterization, token IDs, Unicode IDs,
OCR, character labels, output vocabulary, discrete codebook, candidate
classifier, or external language model. Evaluator labels rank continuous
image-derived states only and never enter training or inference. V16 still has
no pixel actuator.

## Memory Architecture

Let `z_1:t` be retinal states and `h^R_1:t` the pretrained recurrent states.
V16 initializes

\[
u^{(0)}_t=h^R_t+W_z z_t.
\]

Three causal memory blocks operate at dilations `1`, `2`, and `4`. Each block
contains:

1. a left-padded depthwise temporal convolution for local visual structure;
2. gated local mixing;
3. global causal scaled-dot-product attention; and
4. a gated feed-forward residual.

The memory produces a bounded correction to the proven V15 base,

\[
h_t=h^R_t+\sigma(g)\,W_o\,\operatorname{LN}(u^{(3)}_t),
\]

where the per-channel gate starts at `g=-4`. The selected checkpoint has mean
`sigmoid(g)=0.018023` (`0.017968` to `0.018085`). Because the gate changed only
slightly and the output projection can rescale its correction, this receipt
does not attribute the frozen gain to a large gate movement. A direct ablation
on a future preregistered bank is required.

## Preregistered Selection

The continuation resumed V15 step 2,000. Before reading the frozen benchmark,
the checkpoint rule was fixed as:

1. maximize development `proposal_anchor_full_top1`;
2. require full context to exceed the last-image branch;
3. require positive normalized context log-probability gain; and
4. require proposal target cosine above `0.10`.

| Step | Full top-1 | Last-only | Context gain | Target cosine | Eligible |
|---:|---:|---:|---:|---:|:---:|
| **2,200** | **6.778%** | 4.429% | +0.07671 | 0.14226 | **yes** |
| 2,400 | 5.839% | 4.491% | +0.07245 | 0.13710 | yes |
| 2,600 | 5.327% | 4.386% | +0.08261 | 0.13615 | yes |
| 2,800 | 6.218% | 6.400% | +0.07396 | 0.14157 | **no** |
| 3,000 | 5.838% | 5.124% | +0.05811 | 0.13520 | yes |
| 3,200 | 4.361% | 3.139% | +0.05798 | 0.12881 | yes |

Step 2,200 was therefore selected before the frozen evaluator was run. The
frozen bank was queried once for this V16 selection.

## Frozen Evaluation

The evaluator uses 512 forms, four independent font views, 1,788 eligible
contexts, 16 continuous samples per context, and eight intrinsic spherical
integration steps. Its bank hash is:

```text
543987ddd754135cff34dbee0f6f91ce1971741d0807754c65650031121181c4
```

| Proposal metric | V15 step 2,000 | V16 step 2,200 | Change |
|---|---:|---:|---:|
| Full-context top-1 | 5.872% | **6.264%** | +0.392 pp / +7 contexts |
| Last-image top-1 | 4.418% | **3.971%** | -0.447 pp |
| Full-minus-last top-1 | +1.454 pp | **+2.293 pp** | +0.839 pp |
| Full-context top-5 | 12.192% | 11.298% | -0.895 pp |
| Normalized context log-probability gain | +0.07069 | **+0.07732** | +0.00662 |
| Target cosine | **0.14557** | 0.13413 | -0.01144 |

| State-flow metric | V15 step 2,000 | V16 step 2,200 | Change |
|---|---:|---:|---:|
| Full-context top-1 | 3.412% | **3.691%** | +0.280 pp / +5 contexts |
| Last-image top-1 | 2.685% | 3.244% | +0.559 pp |
| Normalized context log-probability gain | +0.03032 | **+0.03579** | +0.00546 |
| Sampled target cosine | **0.21576** | 0.19506 | -0.02070 |
| Sampled context cosine gain | **+0.08053** | +0.07947 | -0.00106 |

The retina oracle remains **98.546%**, so perception is not the principal
bottleneck. The V16 proposal and state-flow gates pass random, unigram,
last-only, context-use, and continuous target-signal checks. Both remain below
the **13.143%** symbolic bigram, so the language acceptance flag remains false.

## Compute Receipt

| Property | V16 receipt |
|---|---:|
| Total parameters | `16,471,809` |
| Trainable parameters | `15,138,881` |
| Context-memory parameters | `6,001,536` |
| Proposal parameters | `3,547,968` |
| Classifier parameters | `0` |
| Pixel-actuator parameters | `0` |
| Sequence length | `48` writing images |
| Continuation updates | `1,200` |
| Logged continuation time | `2,047.63 s` |
| Throughput | about `92-114` sequences/s |
| Peak allocated CUDA memory | `1.479 GiB` |
| Device / precision | one RTX 4090 / BF16 |

## What Changed Scientifically

The result strengthens the narrow feasibility claim: a causal model can learn
next-writing structure from pixels and continuous image states at very small
consumer-GPU cost. It also weakens two tempting but unsupported shortcuts:

- adding attention does not automatically solve language;
- a higher top-1 alone is insufficient when target cosine and top-5 decline.

The proper next experiment is an attributable memory ablation with paired
per-context predictions, correction-norm receipts, and a new frozen bank.
Concurrently, an isolated state-conditioned pixel actuator can test whether a
continuous visual plan can be rendered and reread without a glyph lookup. The
language core and drawing core must pass independently before closing the
autoregressive image loop.

## Reproduction

The selected training continuation was:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_predictive_visual_field.py \
  --resume artifacts/predictive_visual_field_v15_language_scale/checkpoint_step_0002000.pt \
  --out artifacts/predictive_visual_field_v16_memory_pilot \
  --maximum-steps 3200 --lr-schedule-origin-step 2000 \
  --samples-per-epoch 100000 --validation-samples 512 \
  --batch-size 64 --num-workers 10 \
  --flow-positions-per-sequence 4 --sampled-positions-per-sequence 2 \
  --samples-per-context 2 --sample-steps 4 \
  --context-memory-blocks 3 --context-memory-heads 6 \
  --context-memory-kernel 5 --context-memory-ff-multiplier 2 \
  --context-memory-lr-ratio 2.0 \
  --flow-weight 0.10 --endpoint-weight 0.10 \
  --sampled-identity-weight 0.05 --sampled-endpoint-weight 0.05 \
  --context-advantage-weight 0.05 --context-advantage-margin 0.10 \
  --visual-anchor-bank-size 512 --visual-anchor-views 4 \
  --visual-anchor-positive-similarity 0.80 \
  --visual-anchor-identity-weight 0.10 --visual-anchor-context-weight 0.10 \
  --visual-anchor-context-margin 0.10 \
  --proposal-geodesic-weight 0.25 --proposal-identity-weight 0.50 \
  --proposal-context-weight 1.0 --proposal-anchor-identity-weight 2.0 \
  --proposal-anchor-context-weight 2.0 \
  --dynamics-lr-ratio 0.1 --lr 6e-5 --minimum-lr-ratio 0.10 \
  --warmup-steps 100 --log-every 40 --validate-every 200 \
  --validation-batches 8 --save-every 200 \
  --precision bf16 --device cuda --seed 20260812
```

The single frozen evaluation was:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_predictive_visual_field.py \
  --checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --out artifacts/predictive_visual_field_v16_step2200_eval \
  --device cuda --precision bf16
```

Artifacts are intentionally git-ignored. The tracked implementation,
configuration, selection rule, and this receipt make the run auditable without
committing model weights.
