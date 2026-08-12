# Predictive Visual Field V15: Image-Only Causal State Proof

> Successor: V16 adds residual multiscale causal visual memory and reaches
> 6.264% proposal top-1 on the same frozen bank. See
> [`predictive-visual-field-v16-memory-result.md`](predictive-visual-field-v16-memory-result.md).

Date: 2026-08-12

## Verdict

V15 is the first experiment in this repository to pass every preregistered
**continuous visual-state** gate on a frozen bank while substantially exceeding
an image-only unigram baseline. It does not pass the symbolic bigram gate and it
does not yet render pixels. The correct result is therefore:

> A compact student trained from writing images learned measurable causal
> language structure in a continuous visual space on one RTX 4090. This is a
> state-model proof, not yet a complete image-output language model.

The selected deterministic visual proposal obtains `5.872%` top-1 over 512
independently rendered Chinese forms. The same network restricted to the last
visible image obtains `4.418%`; the corpus unigram obtains `1.734%`; random
dynamics obtain `0.168%`; and a symbolic bigram obtains `13.143%`. Full history
improves normalized target log-probability by `+0.07069` nats.

The stochastic hyperspherical field independently obtains `3.412%` top-1,
versus `2.685%` last-only and `1.734%` unigram, with `+0.03032` nats normalized
context gain. Its sampled-state target cosine is `0.21576`, compared with
`0.03180` for random dynamics.

## Strict Boundary

The learned student receives:

```text
32x32 writing images -> frozen convolutional retina -> continuous visual states
                     -> causal GRU field -> continuous visual proposal / state flow
```

It does not receive token IDs, strings, Unicode IDs, OCR transcripts, character
labels, target indices, a finite visual codebook, a candidate classifier, or an
external language model. Offline strings select public-domain passages and
render training images. Evaluator labels are used only after inference to score
the continuous states against an external image bank.

The 512-object multiview anchor bank is training-only and is not deployed. Its
positives are inferred from frozen retinal cosine similarity between images,
not from class IDs supplied to the model.

## Model

For writing fixation \(x_t\), the frozen retina produces a unit vector

\[
z_t=R(x_t)\in\mathbb S^{191}.
\]

A three-layer causal visual field integrates the image history,

\[
h_t=G(h_{t-1},z_t),\qquad h_t\in\mathbb R^{384}.
\]

V14 introduced a deterministic **visual proposal**

\[
\mu_t=\frac{P([h_t,z_t])}{\lVert P([h_t,z_t])\rVert_2},
\qquad \mu_t\in\mathbb S^{191},
\]

which predicts the most useful next image-derived state without selecting a
character ID. In parallel, a conditional vector field models uncertainty over
the same sphere. For target \(z\), random source \(\epsilon\), geodesic
\(q_\tau\), and Riemannian velocity \(u_\tau\),

\[
\mathcal L_{\mathrm{sphere}}
=\mathbb E\left\|v_\eta(q_\tau,\tau,[h_t,z_t])-u_\tau\right\|_2^2.
\]

Sampling integrates the tangent field with the spherical exponential map. The
evaluator scores arbitrary candidate images by kernel density in retinal space;
there is no learned 512-way output layer.

The proposal and stochastic field solve complementary problems. The proposal
is a low-variance linguistic state estimate. The field preserves a distribution
over visually plausible next states. A future pixel actuator will convert a
sampled state to ink and reread the result, but no actuator is used in this
experiment.

## Training Receipt

| Property | V15 receipt |
|---|---:|
| Total parameters | `10,470,273` |
| Trainable parameters | `9,137,345` |
| Frozen retina parameters | `1,332,928` |
| Visual proposal parameters | `3,547,968` |
| Classifier parameters | `0` |
| Pixel actuator parameters | `0` |
| Sequence length | `48` image fixations |
| Training records | `6,812` |
| Validation records | `203` |
| Training anchor objects/views | `512 / 4` |
| Terminal step | `3,000` |
| Cumulative V14+V15 time | `1,895.24 s` |
| Observed throughput | about `100-106` sequences/s |
| Peak allocated CUDA memory | `1.181 GiB` |
| Device/precision | one RTX 4090 / BF16 |

V15 resumed V14 step 1,100 and strengthened proposal identity and context
supervision. Step 2,000 was fixed for the external evaluation at the first
stable joint development peak: proposal-anchor top-1 `6.280%`, last-only
`4.559%`, and target cosine `0.1524`. The frozen evaluator was run once on that
checkpoint. Later development checkpoints were not substituted after seeing
the frozen result.

## Frozen Evaluation

The evaluator uses 512 common Han forms, four independent font views, 3,000
sampled events, and 1,788 eligible next-image contexts. V14 and V15 use the same
bank SHA-256:

```text
543987ddd754135cff34dbee0f6f91ce1971741d0807754c65650031121181c4
```

| Branch | Random | Unigram | Last image | Full history | Bigram |
|---|---:|---:|---:|---:|---:|
| V14 proposal top-1 | `0.168%` | `1.734%` | `2.237%` | `2.517%` | `13.143%` |
| **V15 proposal top-1** | `0.168%` | `1.734%` | `4.418%` | **`5.872%`** | `13.143%` |
| V14 state-flow top-1 | `0.168%` | `1.734%` | `1.734%` | `2.349%` | `13.143%` |
| **V15 state-flow top-1** | `0.168%` | `1.734%` | `2.685%` | **`3.412%`** | `13.143%` |

| Diagnostic | V14 | V15 |
|---|---:|---:|
| Proposal full-history log-probability gain | `+0.02853` | **`+0.07069`** |
| Proposal target cosine | `0.22453` | `0.14557` |
| State-flow full-history log-probability gain | `+0.02759` | **`+0.03032`** |
| State-flow sampled context cosine gain | `+0.07000` | **`+0.08053`** |
| State-flow sampled target cosine | `0.20862` | **`0.21576`** |
| Retina oracle top-1 | `98.546%` | `98.546%` |

The lower V15 proposal target cosine is not contradictory: stronger contrastive
training spreads predictions among visual objects instead of regressing toward
the mean. Retrieval, rank, and normalized context probability improve sharply.

## What V8-V15 Taught Us

| Version | Intervention | Frozen or development result | Decision |
|---|---|---:|---|
| V8 | Euclidean state flow | `1.007%` full, `1.119%` last | Reject: no history advantage |
| V9 | Unit-sphere path | `0.447%` full | Reject: outputs concentrate |
| V10 | Image-only anchors | `1.790%` development top-1 | State signal appears; not a frozen language proof |
| V11 | Intrinsic tangent-energy scaling | `4.33%` sampled target cosine on development batch | Geometry corrected; identity remains weak |
| V12 | Sampled geodesic endpoint loss | `0.391%` frozen top-1, target cosine `0.297` | Reject: visual averaging without object separation |
| V13 | Joint anchor and geodesic loss | `1.454%` full, `2.069%` last | Reject: context hurts rank |
| V14 | Deterministic visual proposal plus state flow | proposal `2.517%`; flow `2.349%` | First frozen state/proposal gate pass |
| **V15** | Strong visual identity and causal proposal scaling | proposal **`5.872%`**; flow **`3.412%`** | State proof strengthened; bigram gate still fails |

The decisive innovation is not “diffuse a page harder.” It is to factor the
problem into three continuous visual computations:

1. **Perception:** map arbitrary writing pixels to a metric retinal manifold.
2. **Language:** predict a low-variance next visual state and a distribution of
   alternatives from causal visual history.
3. **Actuation:** render the selected state as pixels and verify it by rereading.

V8-V13 show why pure flow is insufficient: a generative field can approach a
visual average without learning discrete-looking linguistic alternatives. V14
and V15 show that a deterministic continuous proposal supplies an efficient
linguistic mode without reintroducing tokens. The stochastic field can then be
reserved for uncertainty and form variation rather than carrying all identity
and typography in one denoising operation.

## Claims Allowed

The evidence supports these claims:

- strict image-only causal state learning is possible in a 10.47M-parameter
  model on one consumer GPU;
- full visual history improves both rank and normalized target probability;
- a continuous visual proposal can outperform random, unigram, and last-only
  visual baselines without a class vocabulary; and
- hyperspherical flow produces target-related stochastic visual states and
  benefits from history.

The evidence does not support these claims:

- general language understanding;
- readable generated text images;
- historical etymology question answering;
- performance above a symbolic bigram;
- parity with Qwen, GPT, Llama, or any billion-parameter LLM; or
- end-to-end efficiency over token language models.

## Next Falsifiable Milestone

The next language experiment should replace the GRU bottleneck with a compact
multiscale causal visual memory while preserving the frozen retina, proposal,
flow, bank, and evaluation protocol. It must exceed the `13.143%` bigram without
labels entering the student.

Only after that gate should a state-conditioned pixel actuator be trained. Its
first isolated test is not aesthetic quality: generated pixels must reread to
the requested continuous state across fonts and unknown forms. The complete ILM
gate then requires readable autonomous image continuation under write-reread
feedback.

## Reproduction

The model and evaluator are implemented in:

- `ilm/visual_lm/predictive_visual_field.py`
- `scripts/train_predictive_visual_field.py`
- `scripts/eval_predictive_visual_field.py`

The exact V15 continuation command was:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_predictive_visual_field.py \
  --resume artifacts/predictive_visual_field_v14_proposal_pilot/checkpoint_step_0001100.pt \
  --out artifacts/predictive_visual_field_v15_language_scale \
  --maximum-steps 3000 --samples-per-epoch 100000 --validation-samples 512 \
  --batch-size 64 --num-workers 10 --flow-positions-per-sequence 8 \
  --sampled-positions-per-sequence 4 --samples-per-context 2 --sample-steps 4 \
  --flow-weight 0.25 --endpoint-weight 0.25 \
  --sampled-identity-weight 0.10 --sampled-endpoint-weight 0.10 \
  --context-advantage-weight 0.10 --context-advantage-margin 0.10 \
  --visual-anchor-bank-size 512 --visual-anchor-views 4 \
  --visual-anchor-positive-similarity 0.80 \
  --visual-anchor-identity-weight 0.25 --visual-anchor-context-weight 0.25 \
  --visual-anchor-context-margin 0.10 \
  --proposal-geodesic-weight 0.25 --proposal-identity-weight 0.50 \
  --proposal-context-weight 1.0 --proposal-anchor-identity-weight 2.0 \
  --proposal-anchor-context-weight 1.0 --dynamics-lr-ratio 1.0 \
  --log-every 40 --validate-every 200 --validation-batches 8 \
  --save-every 200 --precision bf16 --device cuda --seed 20260812
```

The selected frozen evaluation command was:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_predictive_visual_field.py \
  --checkpoint artifacts/predictive_visual_field_v15_language_scale/checkpoint_step_0002000.pt \
  --out artifacts/predictive_visual_field_v15_step2000_eval \
  --device cuda \
  --precision bf16
```

Generated checkpoints and evaluator artifacts are intentionally git-ignored.
