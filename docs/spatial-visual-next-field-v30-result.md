# Spatial Visual Next-Field V30: Development Result

Date: 2026-08-13

## Verdict

V30 is **rejected as a selected visual-language mechanism** under its
preregistered development protocol. The frozen partition remains sealed and
writer training is not authorized.

This is a completed matched-control experiment, not a broken run. Both
parameter-identical arms start from byte-identical initialized tensors,
complete 8,000 finite BF16 updates, preserve an image-only student boundary,
write clean final checkpoints, and pass their shared integrity controls. The
global control passes all `12/12` integrity gates.

The spatial route learns an alignment-sensitive candidate score, but not useful
conditional language. On 2,048 natural development windows, its 1,024-way
top-1 is `1.2695%`, below the parameter-identical global control (`2.4902%`),
image unigram (`1.3184%`), and symbolic bigram (`11.7188%`). On 512
cross-record pairs with pixel-identical final four images, spatial full-context
assignment is `50.0488%`, versus `50.5859%` for the global control.

Reversing the candidate's `4 x 4` patch order changes spatial scores and drops
natural top-1 to `0.0488%`, but the target log-probability gain is only
`0.015412` nat against a fixed `>0.05` gate. Pair assignment improves by only
`0.3906` percentage point over the same intervention, against `>5` points.
Spatial sensitivity therefore does not establish context-to-target binding.

The spatial arm passes `12/18` common mechanism gates, the control passes
`12/12` integrity gates, the matched comparison passes `5/9` gates, and the
spatial route passes `0/8` language gates.

![Measured V30 result](../publication/ilm-image-native/figures/spatial_visual_next_field_v30_result.png)

## Fixed System

The authoritative model input remains an ordered volume of writing images:

```text
natural sample:  B x 65 x 1 x 32 x 32
context:         B x 64 x 1 x 32 x 32
exact suffix:    B x  4 x 1 x 32 x 32
candidate:       arbitrary 1 x 32 x 32 image
primary output:  B x 16 x 192 continuous next-image field
```

Reading time is the second axis. A `32 x 32 x N` volume or an invertible 2D
page fold is another view of these ordered visual cells, not a character-code
channel. V30 does not train a page fold, 3D convolution, movie model, visual
tokenizer, or writer.

The student receives floating image tensors only. It receives no string,
byte, token ID, Unicode ID, character ID, OCR transcript, glyph lookup,
discrete codebook, vocabulary embedding, vocabulary logits, symbolic n-gram,
or external language-model state. The 1,024-image candidate bank is a
host-side training/evaluation contrast set. It is absent from model state and
checkpoints and is not required by deployed field prediction.

Both arms load the same frozen V16 retina and V28 semantic adapters plus the
same trainable V29 causal context field. V29's candidate-query critic is
discarded. For context images `X`, the common decoder emits one normalized
continuous vector at every location in a `4 x 4` field:

\[
P_\theta(X)\in\mathbb R^{16\times192}.
\]

The spatial route encodes an arbitrary candidate image `y` as aligned frozen
retinal cells `u_p(y)` and scores corresponding locations before reduction:

\[
s_{sp}(X,y)=\tau\frac1{16}\sum_{p=1}^{16}
P_{\theta,p}(X)^\top u_p(y).
\]

The matched global control repeats the candidate's frozen semantic vector at
all 16 locations:

\[
s_{gl}(X,y)=\tau\frac1{16}\sum_{p=1}^{16}
P_{\theta,p}(X)^\top z(y).
\]

The arms have exactly the same architecture, parameters, initialized values,
examples, candidate columns, loss terms, optimizer, schedule, update count, and
score reduction. Only the route-specific frozen candidate representation
differs. The spatial intervention reverses all 16 candidate locations. The
global control is exactly invariant because all its candidate rows are equal.

Each model has `18,641,153` parameters, of which `17,011,777` are trainable.
The two fixed arms run sequentially on one RTX 4090 and take `2,693.27` seconds
and `3,789.99` seconds. The joint audit takes `39.33` seconds, for
`6,522.59` seconds (`108.71` minutes) total wall time. Peak allocated memory is
`1.595290 GiB` for the spatial arm and `1.597071 GiB` for the global control.
These are resource measurements for a rejected experiment, not evidence of
capability-normalized efficiency over a token language model.

## Fixed Receipts

- 7,017 public-domain Chinese records from 16 Wikisource works;
- 6,608 training, 190 development, and 219 sealed frozen identifiers;
- 32,768 deterministic training suffix-4 pairs;
- 2,048 natural development windows and 512 development suffix-4 pairs;
- corpus SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`;
- V29 source checkpoint SHA-256:
  `a8ec991968b577518d801090f5953406de13c688552107f26ac400fc2d508b8a`;
- frozen V16 retina SHA-256:
  `90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe`;
- protocol SHA-256:
  `81d2b2af1eb3a305b4acd1028c004ddddc607e826eea1d50b6d137d32ed180a5`;
- common initialized state SHA-256:
  `507d875d5b14c612bc7c14b0491bcfe5273ae5cf0bf87d0b819d2c2d9547ca8e`;
- spatial final checkpoint SHA-256:
  `11a3a7e9f13f1db932dcc913e1c79b0e0db49b95bd49f98a878897028bf86130`;
- global-control final checkpoint SHA-256:
  `66378d4b972702490f6819d87d95c2576546e15e6fc74d10542307aaf4483411`;
- development audit SHA-256:
  `2d0a3a08e2f5d4b267e276b695448cc8311687a822776a33c0a883dc0a74fd8f`;
  and
- comparison receipt SHA-256:
  `98865b1d25f80079b998dd9c07bdfb7ada0b5701d2814a2604d4911a89eb0419`.

The initialized states have identical keys, shapes, dtypes, values, and state
hash. Final parameter counts are identical. Source, corpus, candidate-bank,
pair, font, audit-window, and rendered-pixel receipts match. Both final
checkpoints exclude optimizer and RNG state, candidate images, form labels,
and training examples.

## Natural-Language Evidence

The evaluator-only candidate bank never enters the deployed model. Unigram,
bigram, and trigram controls are host-only diagnostics built from the training
partition.

| Development measure | Spatial | Global control | Fixed spatial requirement |
|---|---:|---:|---:|
| Full-context top-1 | `0.012695` | `0.024902` | `>=0.15` |
| Full-context top-5 | `0.027344` | `0.046875` | diagnostic |
| Suffix-4 top-1 | `0.006348` | `0.015625` | diagnostic |
| Shuffled-prefix top-1 | `0.000488` | `0.001465` | diagnostic |
| Spatially permuted top-1 | `0.000488` | `0.024902` | diagnostic |
| Image-unigram top-1 | `0.013184` | `0.013184` | control |
| Symbolic-bigram top-1 | `0.117188` | `0.117188` | benchmark |
| Symbolic-trigram top-1 | `0.200195` | `0.200195` | diagnostic |
| Full minus suffix top-1 | `+0.006348` | `+0.009277` | `>0.03` spatial |
| Full minus shuffle top-1 | `+0.012207` | `+0.023438` | `>0.03` spatial |
| Full minus unigram top-1 | `-0.000488` | `+0.011719` | `>0.03` spatial |
| Full minus bigram top-1 | `-0.104492` | `-0.092285` | `>0.01` spatial |
| Full minus bigram log probability | `-1.903108` nat | `-1.712706` nat | `>0.05` spatial |
| Full minus suffix log probability | `+0.021568` nat | `+0.076863` nat | diagnostic |
| Full minus shuffle log probability | `+0.315340` nat | `+0.650060` nat | `>0.03` spatial |
| Full minus spatial permutation log probability | `+0.015412` nat | `0.0` nat | `>0.05` spatial |
| Candidate cross-font identity top-1 | `0.963379` | `0.964355` | `>=0.95` |

The spatial model detects ordering: full target log probability exceeds the
shuffled-prefix control by `0.315340` nat. It also detects local alignment in
raw score and rank. Neither effect becomes a useful conditional distribution.
The position-blind global control has better full top-1 and target log
probability, and it obtains a larger order effect. A simple symbolic bigram is
more than nine times as accurate as the spatial route.

## Pixel-Identical Suffix Intervention

The primary binding audit uses cross-record context pairs whose final four
writing forms and rendered pixels are exactly equal while their earlier
histories and next images differ. Candidate order is independently randomized
in two unseen development fonts.

| Suffix-4 pair measure | Spatial | Global control | Fixed spatial requirement |
|---|---:|---:|---:|
| Suffix pixel equality | `1.0` | `1.0` | exact |
| Suffix score-row maximum error | `0.0` | `0.0` | `<1e-6` |
| Full arm accuracy | `0.500488` | `0.505859` | `>0.65` |
| Full both-correct | `0.050781` | `0.081055` | `>0.40` |
| Suffix-only arm accuracy | `0.500000` | `0.500000` | exact chance |
| Shuffled-prefix arm accuracy | `0.492920` | `0.491943` | control |
| Spatially permuted arm accuracy | `0.496582` | `0.505859` | control |
| Full minus shuffle accuracy | `+0.007568` | `+0.013916` | `>0.10` spatial |
| Full minus shuffle mean margin | `+0.010235` | `+0.034522` | `>0.05` spatial |
| Full minus spatial permutation accuracy | `+0.003906` | `0.0` | `>0.05` spatial |
| Candidate-column permutation error | `0.0` | `0.0` | `<1e-5` |
| Candidate-column accuracy agreement | `1.0` | `1.0` | exact |

The candidate pixels are visible and candidate-column equivariance is exact.
The spatial intervention changes local compatibility values, but it improves
pair correctness by less than one percentage point. The spatial full result is
also `0.5371` percentage point below the global control. This is not a learned
context-to-candidate binding effect.

## Matched-Arm Decision

| Matched measure | Measured spatial minus global | Fixed requirement | Result |
|---|---:|---:|:---:|
| Full pair arm accuracy | `-0.005371` | `>0.05` | fail |
| Pair both-correct | `-0.030273` | `>0.05` | fail |
| Natural full top-1 | `-0.012207` | `>0.01` | fail |
| Natural target log probability | `-0.190402` nat | `>0.05` | fail |

All five matched setup gates pass: initialized values, final parameter counts,
source/data receipts, rendered audit pixels, and completion of 8,000 finite
updates are exact. All four matched capability gates fail, and every measured
difference favors the global control.

## Why The Deterministic Field Fails

The V30 spatial score is linear in the frozen candidate field:

\[
s_{sp}(X,y)=\left\langle P_\theta(X),U(y)\right\rangle.
\]

Positive field regression pushes `P_theta(X)` toward a conditional mean in
frozen retinal space. That is a useful optimization signal for predictable
stroke layout, but natural next writing is multimodal. Different legitimate
next forms can pull the single predicted field toward an average that matches
general visual structure without identifying the history-specific target. The
1,024-way and exact-pair losses were intended to prevent that collapse, but the
fixed audit shows they do not overcome the single-field bilinear bottleneck.

This diagnosis is bounded by the evidence. The run does not prove that every
spatial or continuous visual distribution must fail. It proves that this
single deterministic `4 x 4 x 192` field, decoder, and loss do not outperform a
matched position-blind semantic vector on the fixed natural-Chinese benchmark.

## What The Result Means

V30 establishes six bounded facts:

1. **A candidate-independent spatial next-image field is executable.** The
   compact student emits `[B,16,192]` from image context without a bank.
2. **Local candidate geometry is visible.** Frozen aligned-field cross-font
   identity is `96.3379%`, and patch reversal changes spatial scores.
3. **Visibility and sensitivity are not binding.** Exact-suffix assignment is
   effectively chance, and intervention gains miss their fixed thresholds.
4. **The matched global route is stronger.** It wins all four direct
   capability comparisons despite having identical trainable tensors.
5. **Natural order is not enough.** Both routes assign more probability under
   ordered than shuffled history, while both remain far below a bigram.
6. **Low resource use is not an efficiency result.** About `1.60 GiB` peak
   memory accompanies a rejected mechanism, not comparable language ability.

The result rejects this deterministic next-field mechanism. It does not reject
the image-native language goal, arbitrary visual forms, continuous output, or
the ordered visual-cell interface.

## Next Controlled Question

Any V31 proposal should address **multimodal continuous next-field density**
without adding a writer, page geometry, hidden character channel, or larger
model. A bounded candidate is a small candidate-independent set or continuous
score model over retinal fields, with a matched global-density control and the
same exact-suffix, shuffle, image-frequency, symbolic-bigram, candidate-column,
and patch-permutation audits.

The key requirement is not prettier reconstruction. It is a held-out increase
in the probability and rank of the correct arbitrary image for the correct
earlier context. The field distribution must beat both the global visual
control and symbolic bigram before any page fold, 3D stream, historical-glyph
answer task, or writer is promoted.

This direction requires a separate preregistration. V30 does not authorize it.

## Reproduction

Run the two fixed evidence arms sequentially:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python scripts/train_spatial_visual_next_field_v30.py \
  --route spatial-field --device cuda:0

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python scripts/train_spatial_visual_next_field_v30.py \
  --route global-control --device cuda:0
```

Run the fixed joint development audit:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python scripts/eval_spatial_visual_next_field_v30.py --device cuda:0
```

Regenerate the checked result figure:

```bash
python publication/ilm-image-native/generate_v30_result_figure.py
```

Primary local receipts:

- `artifacts/spatial_visual_next_field_v30_spatial_evidence/checkpoint_final.pt`;
- `artifacts/spatial_visual_next_field_v30_global_control_evidence/checkpoint_final.pt`;
- `artifacts/spatial_visual_next_field_v30_evidence/development_audit.json`;
  and
- `artifacts/spatial_visual_next_field_v30_evidence/comparison_receipt.json`.

Model and audit artifacts remain ignored by Git. The implementation,
preregistered protocol, tracked result receipt, and evidence-derived figure are
versioned.
