# V40 Cross-Form Glyph Factorization Protocol

Status: preregistered development protocol. V40 is a glyph representation and
rendering mechanism, not evidence of language understanding.

## Question

Can a small image-only model improve the invariance of a qualified continuous
glyph codec across historical forms without losing the ability to render the
visible form directly?

The intervention is explicit:

- content is what remains stable for the same character family across different
  historical stages;
- form is what remains stable for different character families sampled from the
  same historical stage;
- a content state and a form state must recombine into V34 surface geometry and
  then into pixels through the frozen V34 decoder.

Character-family and stage labels are offline training/evaluation supervision.
They are never model inputs and are unavailable at deployment.

## Fixed evidence sources

- V34 EMA checkpoint SHA-256:
  `a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33`
- Historical SQLite SHA-256:
  `c563e8587d7dcacf73704c0fb7816f6d830db11122e0a3da62678b3a7119f738`
- Historical raster manifest SHA-256:
  `3c4064441563c88dffe0c36d42cce0c381bf8b401b764b87484edfb4aa7db99c`
- Family partition: the existing SHA-256 V34 train/development/sealed split.
- Development sampler seed: `20264001`.
- Production architecture: 192 continuous content dimensions, 64 continuous
  form dimensions, and 7,530,241 trainable parameters. V34 remains frozen.

The model receives four `1 x 32 x 32` raster tensors: an anchor, another stage
of the same family, and two same-stage references from different families. It
does not receive strings, Unicode values, token IDs, character IDs, stage IDs,
OCR output, a vocabulary, or a codebook.

## Baselines

Every evaluation must report, on the exact same sampled pairs:

1. frozen V34 surface-latent cross-era retrieval;
2. zero-trained V40 content retrieval;
3. zero-trained V40 self and external-reference raster metrics;
4. the trained warm-start EMA route.

The earlier 299-family diagnostic found 45.82% top-1 retrieval versus 0.334%
chance for one fixed pair sample. It is context, not a substitute for the
exact-pair baseline above.

## Bounded pilot

- 250 updates, batch size 128, seed `20264000`;
- AdamW, peak learning rate `3e-4`, 100 warmup updates, cosine decay;
- evaluate all development families at updates 0 and 250;
- no sealed evaluation and no downstream causal writer attachment.

Continue to the 3,000-update run only if all conditions hold:

- all losses, gradients, parameters, and reports are finite;
- content top-1 is at least the exact-pair frozen-surface top-1;
- content top-1 improves by at least 3 percentage points over zero-trained V40;
- external-reference ink F1 improves by at least 0.15 over zero-trained V40;
- the form-stage margin improves by at least 0.05 over zero-trained V40.

Failure means diagnose or redesign V40. It does not justify extending the pilot
or evaluating the sealed split.

## Development qualification

The independent 3,000-update run qualifies on development only if:

- content top-1 is at least 0.60;
- content top-1 exceeds the exact-pair frozen-surface baseline by at least 0.10;
- content MRR is at least 0.70;
- external-reference raster ink F1 is at least 0.70;
- self-reconstruction raster ink F1 is at least 0.75;
- same-stage/different-family form cosine exceeds
  different-stage/same-family form cosine by at least 0.10.

Only after all development gates pass may one evaluate the untouched sealed
families. The sealed gate is content top-1 at least 0.55, content MRR at least
0.65, external-reference ink F1 at least 0.65, and positive form-stage margin.

## Claim boundary

Passing V40 would establish a compact cross-form visual representation and a
conditioned glyph renderer on held-out character families. It would not by
itself establish lexical knowledge, syntax, semantics, factual recall,
instruction following, page-level reading, or language generation. Those
claims require the later causal glyph-motor and semantic-planner gates.
