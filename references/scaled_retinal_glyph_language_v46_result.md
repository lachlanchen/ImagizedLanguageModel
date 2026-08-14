# Scaled Retinal Glyph Language V46 Result

Date: 2026-08-15

Decision: **non-qualifying development result; 10/14 preregistered gates
pass.** V46 preserves the bounded ordered-raster language signal and improves
target log probability over V42, but it does not deliver the required gains in
full top-1 or autonomous generated identity, does not pass counterfactual
binding, and reduces generated pixel F1. The V43 writer and frozen evaluation
partition remain closed.

## Question Tested

V45 qualified a centered, variance-balanced, exactly invertible raster field.
V46 tests the authorized next question: does the exact V42 causal architecture,
trained from initialization in the **full** scaled V45 field, improve natural
Chinese next-raster language and bank-free raster generation?

For signed binary raster DCT field `d`, V45 mean `mu`, matrix-power transform
`A`, and fixed weighted RMS radius

```text
s = 19.622622215774165,
v = A(d - mu) / s.
```

Unlike V45's direction-plus-log-radius reporting view, `v` retains direction
and radius in one continuous 1,024-dimensional field. Its inverse is

```text
d = mu + A^-1(s v).
```

V46 keeps V42's learned-layer shapes and exact parameter count. It adds no
token, Unicode or character ID, OCR system, visual codebook, quantizer, glyph
lookup, deployed candidate bank, or external language model.

## Immutable Receipts

- production implementation commit:
  `fa502ac322a1075ce80b6933b720396bd53b0678`
- protocol SHA-256:
  `1170d254eb825c4ac0c3c651348ab4c82e8de790bd5908a14ec14b9a9e2ca45e`
- corpus manifest SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`
- V45 field checkpoint SHA-256:
  `0e5947d85a8baeff99d92996ee8434d3aceab39e64042c9b6ec1a142aa174534`
- V45 field-state SHA-256:
  `08b57734ac3ded0c1438cc4bf963d92357ce1f1d31ae49ee6548c56c19db019d`
- V46 final checkpoint SHA-256:
  `98e9946340d408030d97eb4a232b0698aa23a06ef698085a0ba6d0bd769ec9b5`
- tracked development report SHA-256:
  `553733028545cd40fcaedfccdfbc57acccdc0c2950251b26039a15090ba08a44`
- tracked generated-pair sheet SHA-256:
  `111d07e788b11427fff5b36618213a6a15c8e0f1cf59fa8fe18b135447d4136c`

The tensor-free report, summary, training summary, and real generated-pair
sheet are in `publication/ilm-image-native/evidence/v46/`. The 300 MB local
checkpoint is identified by digest and is not committed.

## Frozen Production Run

| Item | Measured |
|---|---:|
| trainable parameters | `24,346,497` |
| updates | `10,000` |
| train elapsed | `943.669` seconds |
| fixed audit elapsed | `12.108` seconds |
| total train plus audit | `955.777` seconds |
| peak allocated CUDA memory | `0.64668` GiB |
| device | one RTX 4090 D, GPU 0 |
| maximum FP64 round-trip DCT error | `3.4639e-14` |
| round-trip binary pixel accuracy | `1.00000` |

Every metric is finite. Fixed training and evaluation arguments, parameter
count, source hashes, protocol hash, V45 field receipt, exact inverse, student
boundary, runtime bound, and closed frozen partition all verify.

## Natural Next-Raster Language

| Fixed development metric | V42 | V46 | Decision |
|---|---:|---:|---|
| full ordered top-1 | `0.19971` | `0.20752` | gain `0.00781`; required `>0.01`, fail |
| image-unigram top-1 | `0.01416` | `0.01416` | V46 full gain `0.19336`, pass |
| symbolic-bigram top-1 | `0.12256` | `0.12256` | V46 full gain `0.08496`, pass |
| shuffled-history top-1 | `0.18359` | `0.19238` | V46 full gain `0.01514`, pass |
| full target log probability | `-5.25531` | `-4.93553` | gain `0.31978` nat; pass |
| full minus shuffled log probability | `0.17391` | `0.17146` | required `>0.05`, pass |
| exact-suffix arm accuracy | `0.53027` | `0.54297` | required `>0.60`, fail |

V46 therefore retains evidence that ordered visual history contributes beyond
unigram, bigram, and shuffled controls. The full-top-1 improvement over V42 is
only 0.781 percentage point, below the frozen one-point requirement. The
counterfactual gain is also too small to establish reliable earlier-history
binding.

## Autonomous Raster Generation

Four stochastic full-field candidates are generated without a glyph bank; the
candidate nearest the deterministic anchor in Euclidean field distance is
inverted to pixels and reread by the same raster model.

| Fixed generated audit | V42 | V46 | Decision |
|---|---:|---:|---|
| generated identity top-1 | `0.08203` | `0.08594` | gain `0.00391`; required `>0.01`, fail |
| anchor identity top-1 | `0.19971` | `0.20313` | report only |
| generated pixel F1 | `0.37308` | `0.35943` | required `>0.55`, fail |
| generated blank rate | `0.00000` | `0.00000` | pass |
| generated target cosine | -- | `0.07983` | report only |
| deterministic-anchor radius MAE | -- | `0.75034` | report only |
| selected-sample radius MAE | -- | `0.26379` | report only |
| generated ink-density ratio | -- | `0.91002` | report only |

The real held-out raster sheet confirms the quantitative failure. Some samples
retain recognizable character structure, but many fragment strokes or move
toward a different identity. Stochastic samples recover radius substantially
better than the deterministic anchor, yet selecting by distance to that anchor
does not preserve the anchor's identity accuracy or improve pixel F1.

## Gate Decision

Passed:

- full top-1 beats image unigram;
- full top-1 beats symbolic bigram;
- ordered log probability beats shuffled history;
- ordered top-1 beats shuffled history;
- generated identity beats V46's image unigram;
- generated blank rate is below `0.02`;
- full log probability beats V42 by more than `0.05` nat;
- student image-only boundary is clean;
- peak allocated CUDA memory is below 18 GiB;
- protocol, source, field, inverse, partition, and runtime integrity is clean.

Failed:

- counterfactual full-history arm accuracy: `0.54297 <= 0.60`;
- full top-1 gain over V42: `0.00781 <= 0.01`;
- generated identity gain over V42: `0.00391 <= 0.01`;
- generated pixel F1: `0.35943 <= 0.55`.

An independent calculation from the tracked report reproduces all 14 booleans
and the `10/14` count exactly.

## Diagnosis

V45's field solves a real conditioning problem, and V46 can learn ordered
next-raster probability in that field from scratch. It does not solve the
coupling between three quantities needed for autonomous writing:

1. discriminative next-glyph direction;
2. calibrated full-field radius;
3. a sampled field whose exact inverse is a high-fidelity target raster.

The late-run deterministic anchor radius error rises rather than converges,
while the stochastic energy samples attain much lower radius error. At the same
time, selected generated identity remains far below anchor identity. This is
evidence against treating one isotropic Euclidean energy score over the full
1,024-dimensional field as a sufficient writer objective. It is not evidence
against continuous raster language modeling itself.

## Bounded Conclusion

V46 is a useful negative result. Scaling the qualified V45 field and training
the V42 architecture from scratch preserves ordered visual language evidence
and improves calibrated target log probability, but it does not meet the
preregistered V42 rank gain, binding, or raster-quality requirements. V46 is
not a complete ILM and does not justify opening the V43 writer or frozen split.

The next bounded experiment should keep the continuous image-only boundary and
the fixed V45 transform, but explicitly factor **identity direction**, **ink
mass/radius**, and **spatial residual rendering** during learning and selection.
It must be preregistered against the same V42/V46 controls. Larger scale is not
authorized until that smaller mechanism improves binding and generated pixels
without a candidate bank.
