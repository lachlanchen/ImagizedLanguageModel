# V26 Frozen Visual Compatibility Probe

Date: 2026-08-13

Status: completed post-hoc development diagnostic; not preregistered evidence

## Question

V26 produced materially different history residuals for contexts with the same
four-glyph suffix, but its stochastic particle head ranked the corresponding
next-glyph images at chance. This diagnostic asks whether a small deterministic
visual scorer can recover the missing association from V26's frozen hidden
state.

It tests localization, not model selection. It does not open the frozen split,
train a writer, or revise any V26 gate.

## Controlled Design

The final 8,000-update V26 checkpoint is frozen completely. A three-branch probe
with 1.11M trainable parameters in total gives each scorer one of:

- the last-glyph appearance state;
- the earlier-history residual; or
- the fused V26 state.

Every candidate is an independently rendered next-glyph image encoded by the
frozen retina. The scorer is a normalized, candidate-conditioned visual energy:

\[
s(c,y)=\tau^{-1}\left\langle
\frac{f(c)}{\lVert f(c)\rVert_2},
\frac{g(R(y))}{\lVert g(R(y))\rVert_2}
\right\rangle.
\]

For each suffix-matched pair `(c_A,y_A),(c_B,y_B)`, a symmetric 2-by-2 cross
entropy requires both diagonal assignments. No string, token ID, Unicode ID,
character label, OCR output, or external model reaches the scorer.

Training is one deterministic pass over the same 16,384 cross-record suffix-4
training pairs used by V26. Development uses 512 cross-record suffix-4 pairs
from disjoint records and two unseen development fonts. Each source pair yields
two cross-font assignments and 2,048 scored arms. Shared suffix pixels are
checked exactly.

## Result

| Development measure | Result |
|---|---:|
| Source pairs | `512` |
| Cross-font assignments | `1,024` |
| Scored arms | `2,048` |
| Suffix pixel equality | `1.000000` |
| Appearance-only arm accuracy | `0.500000` |
| History-residual arm accuracy | `0.506836` |
| Fused-state arm accuracy | `0.503418` |
| History mean margin | `0.000108` |
| Fused mean margin | `0.000371` |
| Raw-retina identity control | `0.999512` |
| Raw-retina mean cosine margin | `0.747237` |
| Runtime | `93.46 s` |
| Peak allocated CUDA memory | `2.482 GiB` |

The appearance state is an exact chance control because pair members have
pixel-identical suffixes. History and fused states are also at chance, not above
the exploratory `0.65` threshold. This is not caused by indistinguishable
candidate images: raw frozen-retina cosine assigns the same cross-font target
images correctly on `99.951%` of arms.

## Decision

The V26 stochastic proposal suppressed context, but it was not the only failed
component. A changed V26 history residual does not expose a transferable
next-glyph relation that this deterministic nonlinear scorer can decode after a
complete training-pair pass.

V27 should therefore learn the context representation and candidate-conditioned
visual compatibility jointly. Its primary loss must make the correct next-image
assignment necessary on matched alternatives during training. It should retain
an appearance-only chance control, a raw-retina identity control, exact suffix
and history-shuffle interventions, and natural unigram/bigram comparisons. A
stochastic form writer remains downstream and unauthorized until deterministic
language compatibility passes.

This result does not prove that no decoder could recover any information from
V26, nor does it reject image-native language modeling. It rejects preserving
the V26 context encoder as the default V27 foundation.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/probe_v26_visual_compatibility.py \
  --device cuda \
  --batch-size 32 \
  --steps 512 \
  --num-workers 8 \
  --out artifacts/v26_visual_compatibility_probe
```

Primary ignored receipts:

- `artifacts/v26_visual_compatibility_probe/diagnostic.json`;
- `artifacts/v26_visual_compatibility_probe/probe.pt`; and
- `artifacts/factorized_visual_context_v26_evidence/checkpoint_final.pt`.

The diagnostic JSON SHA-256 is
`a4588daaeb9ca9c5dd2878481363ef48d6e04728621149f31e7333fb13a2306b`.
