# Visual Path Alignment V38: Measured Result

Date: 2026-08-14

Decision: **`not-qualified`**

Sealed split: **unopened (0 rows rendered)**

Renderer status: **not authorized**

## What V38 tested

V38 tests whether paired visual paths can make an image-only reader stable to
font and wording changes while learning a prompt-conditioned answer state. The
90,753,281-parameter student receives only a `3 x 16 x 1024` prompt raster and
a 64-element clean patch mask. It emits 1,024-dimensional continuous prompt
and answer states plus a visual-length estimate. It does not emit a raster.

Five distinct font paths per training item provide clean and augmented prompt,
paraphrase, and answer views. A full answer map replaces V37's angle-limited
residual planner. Detached, semantically near answer targets supply hard
negatives during training, but no target or candidate bank is deployed.

The checkpoint contains no strings, token or Unicode IDs, character IDs, OCR,
vocabulary logits, visual codebook, candidate bank, target tensor, BGE or Qwen
runtime, teacher call, or network client. Evaluator retrieval probes the
continuous states after all student inference; it is not generated text or a
deployed retrieval system.

## External work and independence

Strong external work is welcome when attribution, license, training role, and
runtime status are explicit. It is not silently relabeled as an ILM
contribution.

- Pixel-Linguist-v0 initializes the visual reader through the V37 EMA state.
  Repository revision:
  `086b70818b2241e81b0de131aa5debe982af7a54`; visual-weight SHA-256:
  `84c1bfbeada57e7e70164811a201a116ad18c22df69beb762fdbb853f8e02676`.
  Its upstream weight license is unstated, so derived weights remain local
  research artifacts and are not redistributed.
- BGE-M3 constructs detached prompt and answer targets offline. It is absent
  from the student process and deployed checkpoint.
- Qwen-family models prepare and audit 1,024 training-only paraphrases. The
  pinned 4B generator, 8B judge, and 30B adjudicator are offline data tools,
  not runtime components or project contributions.
- PIXEL, Text Rendering Strategies, PixelGPT, and SPIRAL are prior work that
  motivates rendered language and paired-path invariance. V38 does not claim
  their methods or results as its own.

Here, *independent* means that final inference is self-contained and
image-only. It does not mean every initialization weight or offline training
target was learned from scratch.

## Completed run

| Property | Measured value |
|---|---:|
| Train records | 5,822 |
| Training-only paraphrases | 1,024 |
| Development records | 196 |
| Updates | 8,000 / 8,000 |
| Examples consumed | 512,000 |
| Training time | 5,982.31 s (99.71 min) |
| Peak allocated CUDA memory | 3,188,168,704 B (2.969 GiB) |
| GPU | GPU 0, one NVIDIA GeForce RTX 4090 D |
| Deployable parameters | 90,753,281 |
| Head-realignment prompt/answer rank | 26.56 / 26.34 |
| Final prompt/answer rank probe | 28.00 / 30.45 |
| Checkpoint SHA-256 | `eea1595107455977bc7ffb96dde3d4cda733186f628f46ab4fb789e83802fcde` |
| Standalone EMA SHA-256 | `25e2fd2652db537455eec57502ffe9e4b51c9cf964311d681c7a2b6e429a8429` |

The schedule uses 500 head-realignment updates and 7,500 full-path adaptation
updates, effective batch 64, BF16, 512 detached candidates, and 16 nearest
negatives per positive. All 8,000 updates complete without a stop signal. The
run, optimizer, EMA, checkpoint, and evaluation routes are finite.

## Frozen development result

| Metric | EMA | Raw | Frozen EMA gate |
|---|---:|---:|---:|
| Prompt-state top-1 | 60.71% | 60.71% | at least 55% |
| Prompt-state top-5 | 86.73% | 85.71% | at least 80% |
| Prompt-state MRR | 0.7236 | 0.7226 | recorded |
| Prompt paired cosine | 0.3789 | 0.3791 | at least 0.50 |
| Direct answer-raster top-1 | 44.39% | 43.37% | at least 45% |
| Prompt-conditioned answer top-1 | 21.94% | 21.94% | at least 35% |
| Prompt-conditioned answer top-5 | 49.49% | 49.49% | at least 65% |
| Prompt-conditioned answer MRR | 0.3460 | 0.3436 | at least 0.45 |
| Answer paired cosine | 0.2441 | 0.2428 | at least 0.30 |
| Answer correct-minus-cyclic margin | 0.2050 | 0.2037 | at least 0.15 |
| Answer correct beats cyclic | 83.67% | 83.67% | at least 85% |
| Counterfactual assignment | 89.80% | 89.80% | at least 90% |
| Held-font prompt/answer cosine | 0.7925 / 0.7323 | 0.7944 / 0.7321 | at least 0.75 / 0.75 |
| Paraphrase top-5 | 73.33% | 73.33% | at least 70% |
| Paraphrase prompt/answer cosine | 0.5408 / 0.4908 | 0.5403 / 0.4871 | at least 0.70 / 0.70 |
| Transition-direction cosine | 0.3401 | 0.3392 | at least 0.25 |
| Prompt-answer cosine | 0.5769 | 0.5693 | at most 0.95 |
| Answer-state effective rank | 49.02 | 49.28 | at least 32 and 40% of target rank |
| Visual-length MAE | 3.370 | 3.370 | at most 3 patches |

The primary EMA route passes 25 of 39 conjunctive conditions. Integrity,
runtime boundary, resource, prompt retrieval, intervention drops, held-font
prompt consistency, paraphrase top-5, transition geometry, answer rank, and
other controls pass. Prompt and answer absolute cosine, direct answer reading,
answer retrieval, answer cyclic win, held-font answer transfer, both
paraphrase-consistency measures, and length fail. Counterfactual assignment is
0.898 against a 0.900 gate and therefore also fails; a narrow miss is still a
miss.

Raw weights pass 24 of 39 only because the preregistered primary route is EMA.
Their substantive metrics are nearly identical. EMA lag is not the cause of
the failed answer path.

## What was learned

V38 improves the visual reader. Relative to V37, prompt top-1/top-5 rises from
47.45/77.55 percent to 60.71/86.73 percent and paired prompt cosine rises from
0.2374 to 0.3789. Canonical-to-held-font prompt consistency rises from 0.4130
to 0.7925. The answer transition is no longer an almost exact copy of the
prompt: mean prompt-answer cosine falls from 0.9972 to 0.5769, transition
direction reaches 0.3401, and answer-state effective rank reaches 49.02.

The central answer relation does not generalize enough. Answer top-1 improves
only from 20.41 to 21.94 percent and top-5 from 45.41 to 49.49 percent. Held
font and paraphrase answer consistency remain below gate. The final training
batch reaches 96.35 percent answer-plan top-1 while development remains 21.94
percent. That gap is diagnostic training evidence, not a capability result. It
is consistent with overfitting a 5,822-pair curriculum across roughly 88
effective passes and with compressing a conditional answer into one vector and
one map.

V38 therefore supports a narrower claim: paired visual paths, hard negatives,
and a full answer map materially improve image reading, font invariance, and
transition geometry on one 4090. They do not yet establish a general answer
mechanism, language generation, or image generation.

## Decision and next bounded proof

The exact decision is `not-qualified`. Zero sealed rows were rendered. The
raster renderer remains unauthorized.

The next experiment should retain the successful image-only reader and frozen
audit boundary, but test the diagnosed bottleneck directly:

1. expand the instruction-relation curriculum substantially and reduce repeat
   exposure, with deduplicated operation, topic, length, and answer-form strata;
2. replace one answer vector with a small ordered set of continuous answer
   slots or recurrent latent dynamics, still conditioned only on prompt pixels
   at deployment;
3. train cross-font and paraphrase consistency on the answer dynamics, not only
   the reader state;
4. include held-operation and held-composition development probes so memorized
   pair maps cannot pass;
5. use strong external document or language models for offline preparation
   when provenance and licenses are exact, while keeping them absent from the
   final ILM runtime; and
6. keep sealed evaluation and raster generation closed until every semantic
   gate passes.

## Tracked evidence

Only small immutable receipts and reports are tracked. Weights, target banks,
similarity tensors, model caches, and source books remain outside Git.

| Evidence file | SHA-256 |
|---|---|
| `run_receipt.json` | `8b2a0ea2b5c786d7945e9d60b303d48c9da195d90fee9957b3edb585458df3c8` |
| `training_summary.json` | `59d2e3b9dac3c4fc7d2b03bc3197fa8588ba098703d9c16c67aae01393c74fbe` |
| `development_report_ema_v38.json` | `835cba957dcaa735889ce4408ea60d81d1fe3ac35a42954d31d4ae543ffadfa4` |
| `development_report_raw_v38.json` | `931809bc29b9f7fe8c1432a1488a08e732a51f096a99ef760df42ef122e02eda` |
| frozen protocol | `cc7112e3f04e7fab622652d7da21955a5b632c5008e51f30182f3bcf2094c0c5` |

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_visual_path_alignment_v38.py \
  --device cuda:0 --precision bf16 \
  --out artifacts/visual_path_alignment_v38_20260814

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_visual_path_alignment_v38.py \
  --checkpoint artifacts/visual_path_alignment_v38_20260814/checkpoint_latest.pt \
  --out artifacts/visual_path_alignment_v38_20260814/development_report_ema_v38.json \
  --device cuda:0 --precision bf16 --batch-size 32

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_visual_path_alignment_v38.py \
  --raw-weights \
  --checkpoint artifacts/visual_path_alignment_v38_20260814/checkpoint_latest.pt \
  --out artifacts/visual_path_alignment_v38_20260814/development_report_raw_v38.json \
  --device cuda:0 --precision bf16 --batch-size 32

PYTHONPATH=. python publication/ilm-image-native/generate_v38_result_figure.py
```
