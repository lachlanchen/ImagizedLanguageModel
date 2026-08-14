# Visual Answer Trajectory V39: Exploratory Pilot Diagnosis

Date: 2026-08-14

Status: `mechanism-signal-observed; not qualified`

## Question

The first V39 pilot asked whether a V38-initialized, image-only reader could
learn more than a repeated global-answer prior when trained to emit one global
answer state and sixteen ordered continuous answer-span states. The pilot was
deliberately limited to 512 Chinese instruction records. It did not open the
sealed split or a raster writer.

## Reproducible inputs

- V38 EMA initialization:
  `25e2fd2652db537455eec57502ffe9e4b51c9cf964311d681c7a2b6e429a8429`;
- 512-record train target bank:
  `e1de585fb3ca3a606232ad67f184bffd7974e099e2db1a73132538b9e4a6c348`;
- complete development target bank:
  `830f457ceb2efdcf4340bee1c89b7008e5ce0e572f16b56bf20054970d3536e6`;
- zero-trained V39 checkpoint:
  `eaa286d3d1b3c086441c6de0b945a0c48f688856fc8ff3c0edf9962f7d5ff365`;
- zero-trained development report:
  `f78af63f54d16d6a712f0b4965149e069f5a75d425605c412800a9d42ed83c4d`;
- pilot EMA checkpoint:
  `1f3f226f74d378a514a72eb3868fd40680d32ed0e12603816ec4769e968c3691`;
- pilot resumable checkpoint:
  `35dbb20a2d88d51612da8d3b6793de389797a46756d6c4d1604b1596d17024fd`;
  and
- full pilot development report:
  `2f9fb37b97952ecb96c17da19055fb41b8b0f3535ec8b544b6223fbe5dd19e40`.

The development bank contains 1,544 records and 9,747 ordered answer spans.
The pilot used 50 frozen-reader updates followed by 100 full-path updates,
effective batch 32, BF16, and all-parameter EMA. It completed in 105.46 seconds
with 2.98 GB peak allocated GPU memory on one RTX 4090 D.

## Held-out result

| Metric | Zero-trained V39 | 512-record pilot | Difference |
| --- | ---: | ---: | ---: |
| Global answer top-1 | 0.04210 | 0.04339 | +0.00130 |
| Global answer MRR | 0.08102 | 0.08302 | +0.00201 |
| Global answer cosine | 0.16444 | 0.18048 | +0.01605 |
| Segment top-1 | 0.00564 | 0.00523 | -0.00041 |
| Segment MRR | 0.01221 | 0.01264 | +0.00044 |
| Segment paired cosine | 0.12362 | 0.13623 | +0.01261 |
| Exact position beats next | 0.50445 | 0.51164 | +0.00719 |
| Transition-direction cosine | 0.00000 | 0.00050 | +0.00050 |
| Visual-length MAE | 11.8469 | 9.0253 | -2.8216 |

The global answer top-1 rate is about 67 times the `1 / 1544` chance rate.
The segment top-1 rate is about 51 times the `1 / 9747` chance rate, but much
of that signal already exists when the zero-initialized planner repeats the
V38 global answer state at every position. MRR, cosine, mean rank, and weak
position preference improve after training; exact segment top-1 does not.

## Intervention evidence

The pilot is not merely a position-only predictor. Relative to the correct
prompt, a cyclically shuffled prompt reduces:

- global answer cosine by 0.11028;
- global answer MRR by 0.07609; and
- segment paired cosine by 0.06519.

A blank prompt reduces the same values by 0.15305, 0.07790, and 0.11793. All
states are finite. Canonical-to-held-font cosine is 0.7690 for the reader,
0.7094 for the answer, and 0.6976 for active segment states. Simplified-to-
traditional conversion gives 0.7959, 0.7474, and 0.7309 respectively.

These controls establish a bounded visual-conditioning signal. They do not
establish answer correctness, ordered language generation, or raster output.

## Failures exposed

1. **Ordered progression is too weak.** Exact-position preference is only
   0.5116 and transition-direction cosine is effectively zero.
2. **The trajectory does not yet beat the inherited answer map.** It improves
   long-answer MRR and cosine slightly, but aggregate final-answer MRR remains
   0.00016 below the checkpoint's V38-compatible baseline route.
3. **Discrete stopping is misdecoded.** Thresholding each hazard at 0.5 makes
   every record appear to use all sixteen positions. A bounded hazard process
   defines a categorical first-stop distribution; evaluation and inference
   must use its mode or expectation instead of independent thresholding.
4. **The short EMA is stale.** A fixed 0.999 decay retains about 86% of the
   zero-trained planner after only 150 updates. A bias-corrected warm-start
   decay is needed for short and staged runs.
5. **Semantic states are not a writing code.** BGE targets can test meaning and
   relation, but they cannot uniquely reconstruct arbitrary glyph sequences.

## V39.1 correction before scale-up

The full-data run should preserve the model boundary and change only justified
mechanics:

1. derive normalized first-stop probabilities
   `p_k = a_k sigmoid(s_k)`, forcing the final bounded position to stop;
2. train stop position with per-record categorical negative log likelihood and
   a count-moment loss;
3. compare exact span positions only with real preceding/following positions,
   removing the current last-to-first wraparound negative;
4. use a warm-started all-parameter EMA whose decay approaches, but never
   exceeds, the configured ceiling;
5. increase stop, length, and transition weights modestly; and
6. train on the full 44,637-record bank before deciding whether the semantic
   planner is qualified.

No sealed data or writer is authorized by this pilot.

## Writer path after a semantic pass

V34 already qualifies a 7.4M-parameter, codebook-free `32 x 32` visual codec
for modern and historical writing. V35 showed that predicting those local
latents directly from a prompt fails semantic binding. The justified synthesis
is therefore hierarchical:

```text
prompt raster
  -> V39 global and ordered semantic plan
  -> plan-conditioned causal V34 latent writer
  -> V34 pixel decoder
  -> thresholded visible raster
  -> V34 re-encoder for closed-loop continuation
```

The writer must retain teacher-forced reconstruction, autonomous copy,
correct/shuffled/blank prompt, rereading, and raster-quality gates. It remains
closed until full-data V39 demonstrates materially ordered held-out content.
