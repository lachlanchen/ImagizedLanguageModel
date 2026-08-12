# RFLM V6 Closed-Loop Experiment

Date: 2026-08-12

Status: completed, rejected by the fixed language gate

## Question

The V5 Retinal Flow Language Model could read cross-font Chinese glyph images,
score arbitrary candidate images, and write pixels, but its autonomous output
lost ink and fragmented. The V6 experiment tested one narrow hypothesis:

> Does training on image states sampled by the deployed writer make the visual
> language loop stable and improve next-image prediction without introducing a
> tokenizer, OCR, Unicode IDs, labels, a glyph codebook, or an external LLM?

The intervention held model size, corpus, renderer, and fixed evaluation bank
constant. It changed the state distribution seen during training.

![Measured V5 and V6 autonomous comparison](../publication/ilm-image-native/figures/closed_loop_v6_result.png)

## Student boundary

The learned path remained:

```text
ordered writing pixels -> continuous retina -> recurrent visual state
                       -> continuous pixel flow -> generated writing pixels
                       -> retina reread -> recurrent visual state
```

Offline strings were used only to render the public-domain corpus and to score
the evaluator. The student received no strings, token IDs, Unicode IDs, OCR,
character labels, finite visual vocabulary, inverse glyph table, or external
model call.

## V6 objective

For a clean visual prefix, V6 runs the same sampler, target-retina rereader, and
energy reranker used at deployment. With two sampled candidates and two flow
steps, it selects a bitmap, stops gradients through selection, rereads that
bitmap with the online retina, and advances the GRU. The rollout adds three
losses:

\[
\mathcal L_{\mathrm{roll}} =
0.15\mathcal L_{\mathrm{state}} +
0.35\mathcal L_{\mathrm{energy}}^{\mathrm{roll}} +
0.30\mathcal L_{\mathrm{recovery}}.
\]

`state` aligns clean and induced recurrent trajectories by cosine distance.
`energy` asks the rollout state to select the next real image from a large
continuous candidate bank. `recovery` trains pixel flow toward the clean next
image after a generated prefix. The rollout weight ramps from zero to one over
400 updates. Candidate generation is stop-gradient; learning occurs after the
selected pixels are reread.

This is exact model-induced visual feedback, not generic noise injection. The
shared `sample_visual_candidates()` path is used by training and inference.

## Training receipt

V6 resumed the 3,600-step V5 checkpoint and trained to step 5,200:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_retinal_flow_lm.py \
  --manifest data/visual_grammar/chinese_wikisource_public_domain.jsonl \
  --resume artifacts/retinal_flow_chinese_mvp_v5_multiview_cycle/checkpoint_latest.pt \
  --out artifacts/retinal_flow_chinese_mvp_v6_closed_loop \
  --sequence-length 48 \
  --energy-positions-per-sequence 8 \
  --batch-size 32 \
  --epochs 5 \
  --maximum-steps 5200 \
  --lr 8e-5 \
  --lr-cycle-start-step 3600 \
  --warmup-steps 50 \
  --rollout-start-step 3600 \
  --rollout-ramp-steps 400 \
  --rollout-batch-size 8 \
  --rollout-steps 2 \
  --rollout-candidates 2 \
  --rollout-sample-steps 2 \
  --rollout-guidance-scale 1.5 \
  --precision bf16
```

| Property | Value |
|---|---:|
| Parameters | 11,690,244 |
| Added updates | 1,600 |
| Added training time | 1,606.8 s |
| Peak CUDA allocation | 2.576 GiB |
| Final training throughput | 105.8 sequences/s |
| Rollout state cosine | 0.961 |
| Rollout selected-target cosine | 0.103 |
| Rollout next-image top-1, in-batch | 6.64% |
| Rollout recovery endpoint F1 | 0.695 |

The high state cosine and low selected-target cosine are important together:
V6 learned to recover a clean-like state from its own wrong-looking images, but
the sampled bitmap itself did not become semantically correct.

## Frozen-bank result

All reported checkpoints used the same 512-character, four-view glyph bank:

```text
543987ddd754135cff34dbee0f6f91ce1971741d0807754c65650031121181c4
```

| Measure | V5 step 3,600 | V6 step 4,400 | V6 step 4,800 | V6 step 5,200 |
|---|---:|---:|---:|---:|
| Full-context top-1 | 0.908% | 1.114% | 1.156% | **1.197%** |
| Full-context top-5 | 2.229% | 2.270% | 2.146% | **3.219%** |
| Full-context MRR | 2.450% | 2.542% | 2.635% | **2.899%** |
| Last-fixation top-1 | 1.362% | 1.362% | 1.486% | **1.692%** |
| Retina oracle top-1 | 97.648% | 98.143% | **98.225%** | 98.184% |
| Generated pixel F1 | **0.428** | 0.420 | **0.430** | 0.425 |
| Generated context cosine gain | **+0.0211** | +0.0113 | +0.0068 | +0.0077 |
| Generated sample hit | **1.563%** | 1.042% | 0.000% | 1.042% |

The full-context retrieval trend improved monotonically, but the final model
still lost to last-fixation (`1.692%`), unigram (`1.857%`), and symbolic bigram
(`13.578%`). Generated target signal weakened from V5. The frozen acceptance
result is therefore false.

## Matched autonomous result

Both models received the same rendered prompt, seed `20260812`, eight candidates,
eight sampling steps, guidance `1.5`, soft visual feedback, and 32 requested
cells:

```text
天地玄黃，宇宙洪荒。日月盈昃，辰宿列張。
```

| Autonomous measure | V5 | V6 |
|---|---:|---:|
| Mean ink | 0.2010 | 0.2418 |
| First-eight mean ink | 0.3390 | 0.2321 |
| Last-eight mean ink | 0.1637 | 0.2711 |
| Late/early ink ratio | 0.483 | **1.168** |
| Sparse-cell fraction | 37.5% | **18.75%** |
| Early candidate cosine | 0.549 | 0.443 |
| Late candidate cosine | 0.328 | **0.444** |
| Throughput | 18.9 cells/s | 21.5 cells/s |

V6 no longer exhibits V5's monotonic loss of ink. It sustains character-scale
marks throughout the rollout. Visual inspection still finds no readable Chinese
continuation in either output. Stability improved; language did not emerge.

## Verdict

The experiment supports one claim and rejects a stronger one:

- **Supported:** model-induced visual trajectory training materially improves
  autonomous form stability and modestly improves fixed-bank retrieval.
- **Rejected:** trajectory consistency alone does not make the image writer
  generate the correct next linguistic image or use long visual context.

This is not evidence of Qwen-8B parity, useful Chinese generation, etymology
question answering, or efficiency at matched quality.

## Next falsifiable intervention

The next algorithm should attack the two measured failures directly while
retaining V6's stable feedback loop.

### 1. Visual context-advantage loss

Construct two continuous states for the same target image: `full`, which reads
the complete image prefix, and `last`, which resets memory and reads only the
last fixation. Require the target image to receive a margin from full history:

\[
\mathcal L_{\mathrm{ctx}}=
\max\left(0,
m-s(y\mid h_{\mathrm{full}})+s(y\mid h_{\mathrm{last}})\right).
\]

This uses only images and continuous states. The frozen evaluator already
measures the corresponding hypothesis, so the training objective and acceptance
gate become aligned. A full-state NCE term must remain to prevent the trivial
solution of merely degrading the last-only branch.

### 2. Differentiable sampled-endpoint identity

V6 stops gradients through generated candidate selection. Unroll two flow steps
for one candidate with gradients and contrast its reread endpoint against the
real next-image views. This trains the pixels the model actually samples, rather
than only a one-step denoising estimate:

\[
\mathcal L_{\mathrm{sample}}=
\operatorname{NCE}\left(
R(\hat x_{0}^{K=2}),\{R(y^{(v)})\}_{v=1}^{V}
\right).
\]

Use truncated unrolling on a small rollout subset to preserve the measured
single-GPU budget.

### 3. Keep the V6 stability constraints

Retain stop-gradient rollout state alignment and recovery flow as regularizers.
Do not add a larger model, historical instruction data, or a slow page state
until full context beats the last-only and unigram gates. A slow state cannot be
credited if the existing history is not first shown to carry predictive value.

The next checkpoint must beat V6 on the unchanged bank, restore generated
context cosine gain above the evaluator's `0.02` threshold and `0.01` random
margin, and remain readable for 32 autonomous cells. Otherwise the intervention
is rejected and the recurrent state formulation, not merely its training
schedule, must change.
