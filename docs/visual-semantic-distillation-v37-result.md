# Visual Semantic Distillation V37: Measured Result

Date: 2026-08-14

Decision: **`not-qualified`**

Sealed split: **unopened**

Renderer status: **V37-R remains forbidden**

## What V37 tested

V37 tests whether end-to-end semantic distillation can turn an image-only
reader into a useful continuous answer planner before any raster writer is
trained. The 89,768,706-parameter student receives only a rendered Chinese
prompt and a clean visual patch mask. It emits a 1,024-dimensional prompt
semantic state, a candidate-independent 1,024-dimensional answer plan, and a
visual-length estimate.

The visual encoder is initialized from the external Pixel-Linguist-v0
checkpoint and then adapted end to end. BGE-M3 is used only before training to
construct detached continuous prompt and answer targets. The standalone
student has no BGE or Pixel-Linguist service call, text, token or Unicode IDs,
OCR output, candidate bank, lookup table, visual codebook, or target tensor.
Candidate retrieval exists only in the evaluator and is not generated
language.

## External work and independence

Reusing good external work is intentional when its role and provenance are
exact. V37 does not claim Pixel-Linguist or BGE-M3 as project contributions.

- Pixel-Linguist repository: `Pixel-Linguist/Pixel-Linguist-v0`;
- pinned revision: `086b70818b2241e81b0de131aa5debe982af7a54`;
- visual weight SHA-256:
  `84c1bfbeada57e7e70164811a201a116ad18c22df69beb762fdbb853f8e02676`;
- role: initialization of the student's 12-layer visual encoder;
- BGE-M3 Ollama manifest SHA-256:
  `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`;
- BGE-M3 model-layer SHA-256:
  `daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c`;
- BGE-M3 role: offline detached target construction only.

Here, *independent* means that the trained inference artifact is self-contained
and image-only. It does not mean that every weight was pretrained from scratch.
Pixel-Linguist's released checkpoint states no weight license, so the derived
student remains local-research-only and is not redistributed.

## Completed run

| Property | Measured value |
|---|---:|
| Train records | 5,822 |
| Development records | 196 |
| Updates | 8,000 / 8,000 |
| Examples consumed | 512,000 |
| Training time | 3,695.48 s (61.59 min) |
| Peak allocated CUDA memory | 2,951,057,920 B (2.748 GiB) |
| GPU | GPU 0, one NVIDIA RTX 4090 D |
| Deployable parameters | 89,768,706 |
| Warmup answer-plan effective rank | 9.23 |
| Final answer-plan effective rank | 36.44 |
| Checkpoint SHA-256 | `5bcdbb302645928007a5501fae436cc1f25f22cf4c1646ea650d7c96d3ea0acc` |
| Standalone EMA SHA-256 | `367b0dfd5b54c537aaa4cb41305c6c63c08862921228b706bd1dc37c5c2170f8` |

The schedule used 500 projection-head warmup updates followed by 7,500
full-reader adaptation updates. Effective batch size was 64, with 512 detached
candidate targets sampled per microbatch. All losses, gradients, parameters,
optimizer state, EMA state, targets, checkpoints, and reported predictions
remained finite.

## Frozen development result

| Metric | EMA | Raw | Frozen EMA gate |
|---|---:|---:|---:|
| Prompt-state paired cosine | 0.2374 | 0.2347 | at least 0.70 |
| Prompt-state top-1 | 47.45% | 47.45% | at least 25% |
| Prompt-state top-5 | 77.55% | 77.04% | at least 60% |
| Answer-plan top-1 | 20.41% | 20.41% | at least 30% |
| Answer-plan top-5 | 45.41% | 46.94% | at least 60% |
| Answer-plan MRR | 0.3377 | 0.3340 | at least 0.40 |
| Paired answer cosine | 0.1634 | 0.1621 | at least 0.35 |
| Correct-minus-cyclic margin | 0.1590 | 0.1572 | at least 0.20 |
| Correct beats cyclic | 84.69% | 83.67% | at least 85% |
| Counterfactual assignment | 93.88% | 93.88% | at least 85% |
| Held-font plan cosine | 0.4089 | 0.4088 | at least 0.85 |
| Paraphrase top-5 | 73.33% | 73.33% | at least 50% |
| Original/paraphrase plan cosine | 0.4558 | 0.4523 | at least 0.75 |
| Answer-plan effective rank | 55.37 | 56.83 | at least 32 and 40% of target rank |
| Clean visual-length MAE | 3.682 | 3.678 | at most 3 patches |

The EMA route passes 20 of 33 conjunctive conditions. Exact source and
external hashes, strict tensor mapping, finite checks, model boundary,
parameter cap, completed updates, memory, prompt retrieval, improvement ratio
over the untrained head, counterfactual assignment, paraphrase top-5, and plan
rank pass. Prompt paired cosine, absolute answer retrieval, absolute
improvement, answer cosine, cyclic separation, shuffled/blank cosine margins,
held-font consistency, paraphrase consistency, and length fail.

The answer-plan top-1 gain over the untrained head is 0.19898, narrowly below
the frozen absolute-gain requirement of 0.20. This near miss cannot override
the other failed absolute and invariance gates. Raw weights pass 19 of 33
conditions and do not reveal a hidden qualified route.

## What was learned

V37 fixes two central V36 failures. Clean pre-augmentation masks remove the
length-distribution defect. Its development answer-plan output has effective
rank 55.37 and passes both frozen rank controls; unlike V36's train-target rank
of 4.77/768, collapse is no longer the principal diagnosed bottleneck.
Prompt-state retrieval rises to 47.45 percent top-1, and answer-plan retrieval
rises from V36's 1.02 percent to 20.41 percent. This is a real visual reading
and relational-planning advance.

It is not yet a qualified language mechanism. The plan is directionally useful
but weakly aligned in absolute cosine, changes too much under a held font and
paraphrase, and does not separate the correct answer from difficult alternatives
by the required margin. High counterfactual assignment shows that paired
prompt changes affect plan ordering; it does not establish stable semantics
under surface-form changes.

The direct centered BGE target ceiling on the same 196 pairs is 82.65 percent
top-1, 92.86 percent top-5, and 0.8723 MRR. The semantic geometry therefore
exists in the offline target space. V37 recovers a substantial fraction of it
from pixels, but the remaining gap is in visual-to-semantic invariance and
answer relation transfer, not target availability.

## Decision

V37 is a valid negative result with a positive bounded mechanism. It shows
that a sub-100M image-only student can learn useful visual semantic retrieval
and candidate-free answer planning in about one hour with 2.75 GiB peak
allocated memory on one 4090. It does not prove autonomous language
understanding, answer generation, image generation, or efficiency superiority
over token models.

The complete decision is `not-qualified`. The sealed split stays closed.
V37-R is neither implemented nor trained.

The next bounded experiment should keep the same audited image-only runtime and:

1. train explicit canonical/held-font and original/paraphrase positive pairs,
   rather than relying on incidental augmentation for invariance;
2. use semantically nearest detached negatives and relation-aware objectives to
   improve answer-plan margin without deploying a candidate bank;
3. preserve the successful end-to-end reader adaptation and rank controls;
4. strengthen clean visual-length supervision separately from semantics;
5. allow stronger external visual or document foundations when provenance,
   licensing, training role, and runtime boundary are exact; and
6. retain the same sealed and renderer prohibition until every semantic
   condition passes.

## Tracked evidence

The repository stores only small immutable receipts and reports, not weights,
target tensors, similarity matrices, or model caches:

| Evidence file | SHA-256 |
|---|---|
| `run_receipt.json` | `91738c1c6290bf479be0bc8e92f95142e0fba67f1ce2301de0cac2269d6ad2c7` |
| `training_summary.json` | `6437d7d1fa48e9a32a1badd0f93966145cf756f002867a4f540cb1c8b273826b` |
| `development_report_ema_v37.json` | `5f7941b7fa9668e9fa61abfb6b689073c3a1891977b40250b63561ce88857c7c` |
| `development_report_raw_v37.json` | `aa572525bb3c9697ec37d736960dea59e2b583936e866e87e21cd428ef162fef` |
| frozen protocol | `e3cca1c8eedb387f80a88cf17a93466f59532ea666d6dcbfe57e5d7d5e91f6d7` |

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_visual_semantic_distillation_v37.py \
  --checkpoint artifacts/visual_semantic_distillation_v37_20260814/checkpoint_latest.pt \
  --train-bank artifacts/visual_semantic_distillation_v37_targets/train.pt \
  --development-bank artifacts/visual_semantic_distillation_v37_targets/development.pt \
  --out artifacts/visual_semantic_distillation_v37_20260814/development_report_ema_v37.json \
  --device cuda:0 --precision bf16 --batch-size 32

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_visual_semantic_distillation_v37.py \
  --raw-weights \
  --checkpoint artifacts/visual_semantic_distillation_v37_20260814/checkpoint_latest.pt \
  --train-bank artifacts/visual_semantic_distillation_v37_targets/train.pt \
  --development-bank artifacts/visual_semantic_distillation_v37_targets/development.pt \
  --out artifacts/visual_semantic_distillation_v37_20260814/development_report_raw_v37.json \
  --device cuda:0 --precision bf16 --batch-size 32

PYTHONPATH=. python publication/ilm-image-native/generate_v37_result_figure.py
```
