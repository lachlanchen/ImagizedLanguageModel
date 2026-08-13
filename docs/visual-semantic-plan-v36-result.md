# Visual Semantic Plan V36: Measured Result

Date: 2026-08-14

Decision: **`not-qualified`**

Sealed split: **unopened**

Renderer status: **V36-R remains forbidden**

## What V36 tested

V36-P isolates the semantic step that V35 lacked. A 93,473,281-parameter
student receives only a rendered Chinese prompt and a visual patch mask. It
emits one global and four ordered continuous 768-dimensional plans plus a
visual-length estimate. During training, a frozen Pixel-Linguist-v0 reader
encodes answer images into detached targets. The deployed model and checkpoint
contain no answer teacher, target bank, candidate answers, strings, token or
Unicode IDs, OCR path, character lookup, or visual codebook.

Candidate retrieval exists only inside the evaluator. It measures whether an
autonomously predicted plan selects the paired held-out answer state. It is not
an inference feature and is not generated language.

## Completed run

| Property | Measured value |
|---|---:|
| Train records | 5,832 |
| Development records | 196 |
| Updates | 6,000 / 6,000 |
| Examples consumed | 512,000 |
| Training time | 1,537.63 s (25.63 min) |
| Peak allocated CUDA memory | 1,654,311,424 B (1.541 GiB) |
| GPU | GPU 0, one NVIDIA RTX 4090 D |
| Deployable parameters | 93,473,281 |
| Checkpoint SHA-256 | `e3c1da119616935d045987311123deae2be27c32a719fc28672a07864718153b` |
| EMA artifact SHA-256 | `efe55205d199031afd5bb1b0c9cf010f546000f602bbf2e2c51d2c9a610df50b` |

All training losses, trainable parameters, teacher targets, checkpoints, and
reported predictions remained finite. The final effective-batch top-1 values
were 24.22 percent in stage A and 45.31 percent in stage B. Those are training
batch metrics, not held-out capability.

## Frozen development result

| Metric | EMA | Raw | Frozen EMA gate |
|---|---:|---:|---:|
| Top-1 retrieval | 1.02% | 2.04% | at least 8% |
| Top-5 retrieval | 12.24% | 11.22% | at least 25% |
| Mean reciprocal rank | 0.0783 | 0.0808 | at least 0.15 |
| Correct-minus-cyclic cosine | 0.0427 | 0.0418 | at least 0.05 |
| Correct beats cyclic | 67.35% | 65.82% | at least 70% |
| Counterfactual assignment | 78.57% | 76.53% | at least 70% |
| Held-font prompt-plan cosine | 0.6265 | 0.6100 | at least 0.85 |
| Paraphrase top-5 | 33.33% | 33.33% | at least 20% |
| Original/paraphrase plan cosine | 0.6578 | 0.6473 | at least 0.75 |
| Visual-length MAE | 36.16 | 36.53 | at most 4 patches |

The EMA route passes 13 of 23 conjunctive conditions. Integrity, exact source
and external hashes, strict tensor mapping, model boundary, parameter count,
completed updates, peak memory, shuffled-prompt drop, counterfactual
assignment, paraphrase top-5, and answer-teacher cross-font stability pass.
Absolute retrieval, improvement over the untrained head, cyclic margin, blank
drop, held-font transfer, paraphrase plan consistency, and length fail.

Raw weights do not repair the result. Their 2.04 percent top-1 remains below
the 8 percent gate and only slightly exceeds the 1.53 percent untrained-head
baseline. This rules out EMA lag as the principal explanation.

## What was learned

The counterfactual rate and shuffled-prompt degradation are nonzero evidence
that prompt pixels influence the plan in the intended direction. They do not
establish a useful semantic planner. Correct plans remain poorly separated
from wrong answers, blank input remains too competitive, and font/wording
changes move the plan too far. The full decision must therefore remain
`not-qualified`.

## Post-result diagnosis

This diagnosis uses only the already opened train and development artifacts. It
does not alter the preregistered result.

### Occupancy mask defect

`render_visual_sentence_strip` computes patch occupancy from the raster after
contrast and optional Gaussian noise. Contrast below one shifts white
background below exactly white; the `> 1/255` ink threshold can then mark blank
patches as occupied. Noise has a similar effect.

The resulting target-bank distributions are inconsistent:

| Diagnostic | Train | Development |
|---|---:|---:|
| Mean active answer patches | 37.31 | 11.53 |
| Standard deviation | 26.06 | 5.06 |
| Minimum / maximum | 1 / 64 | 1 / 22 |
| Median answer characters | 19 | 18.5 |

Character lengths are closely matched, so the 3.24x mask-length shift is a
rendering/mask error rather than a linguistic split difference. It directly
explains much of the failed length head and changes which background patches
the frozen teacher pools.

### Low-rank visual target geometry

The normalized train answer targets have mean-vector norm 0.889 and centered
effective rank 4.77 in 768 dimensions. Median pairwise cosine is 0.777 and the
median nearest-neighbor cosine is 0.967. Development targets have effective
rank 19.88, still anisotropic but materially less collapsed. The noisy
train-mask distribution is contributory, but it is not the sole limitation:
the frozen Pixel-Linguist direct prompt-to-answer baseline itself reaches only
4.59 percent top-1 and 13.27 percent top-5 on development.

### External semantic-teacher probe

External work is acceptable when its role and provenance are explicit. A
post-result, nonsealed diagnostic reused the local BGE-M3 embedding model:

- Ollama manifest SHA-256:
  `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`;
- model blob SHA-256:
  `daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c`;
- local server version: Ollama `0.32.6`;
- role: exploratory offline semantic teacher only.

On the same 196 development pairs, direct BGE-M3 prompt-to-answer retrieval is
83.16 percent top-1, 93.37 percent top-5, and 0.8772 MRR. Its development answer
space has centered effective rank 77.27. A train-fitted closed-form ridge map
from clean frozen Pixel-Linguist prompt features to BGE-M3 prompt embeddings
reaches only 2.04 percent top-1 against BGE-M3 answer targets. Therefore the
semantic geometry exists, but it is not linearly recoverable from the frozen
visual representation. The next experiment needs end-to-end visual semantic
distillation and explicit anti-collapse regularization, not another projection
head.

This teacher diagnostic does not qualify V36, does not use sealed data, and
does not place BGE-M3 in the future deployed runtime.

## Decision

V36-P is a valid negative result. It proves that the candidate-free,
image-input continuous planner can be trained cheaply and audited on one 4090,
but it does not prove reliable language understanding or image generation.
The sealed split stays closed. V36-R is not implemented or trained.

The next bounded experiment should:

1. derive occupancy masks from clean text geometry before raster augmentation;
2. distill a strong multilingual semantic space into the image reader using
   prompt and answer images while keeping the teacher offline;
3. adapt enough of the reader to make semantics nonlinear and visually
   recoverable;
4. regularize variance/covariance and measure effective rank;
5. retain blank, shuffled, font, paraphrase, counterfactual, and strict runtime
   boundary controls; and
6. open a direct raster renderer only after the complete semantic gate passes.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_visual_semantic_plan_v36.py \
  --checkpoint artifacts/visual_semantic_plan_v36_20260814/checkpoint_latest.pt \
  --development-bank artifacts/visual_semantic_plan_v36_targets/development.pt \
  --out artifacts/visual_semantic_plan_v36_20260814/development_report_ema.json

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_visual_semantic_plan_v36.py \
  --raw-weights \
  --checkpoint artifacts/visual_semantic_plan_v36_20260814/checkpoint_latest.pt \
  --development-bank artifacts/visual_semantic_plan_v36_targets/development.pt \
  --out artifacts/visual_semantic_plan_v36_20260814/development_report_raw.json

PYTHONPATH=. python publication/ilm-image-native/generate_v36_result_figure.py
```

The external Pixel-Linguist checkpoint states no license. Its weights and the
derived V36 checkpoints remain local-research-only and are not redistributed.
