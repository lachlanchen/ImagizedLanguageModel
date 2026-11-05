# Grounded image-language learning: writing stage

5 November 2026. The complete actor preflight passes at 17.181 GiB; the
separate inherited-context sanity check passes eight fixed text cases. Neither
establishes image-language quality. This first learning stage trains the actual
actor's continuous writer to reconstruct writing from its frozen visual
features. It prepares its output path for subsequent instruction learning;
reconstruction cannot complete the language goal.

## Frozen writing experiment

- Use the existing 27-layer FP32 image encoder and six-layer, 384-wide writer
  from the complete actor. Keep inherited parameters, contextual adapters,
  continuous queries and stopping head unchanged. Only the writer's 19,374,096
  parameters train. No tokenizer, text, character IDs, glyph bank, OCR or model
  service is used by this stage.
- Use the pinned complete-writing training cache, 12,976 examples, report SHA
  `0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e`.
  Each update draws eight Chinese CMRC and eight English Dolly examples,
  uniformly within each source, and one complete 32-by-64 answer block from
  each example. Exclude only the cache's two structural termination blocks.
  White space inside actual answers remains eligible.
- Encode the sixteen independent images together; native image boundaries
  prevent cross-image attention. Each has two 4,096-dimensional output vectors.
  Feed these to the same continuous block writer used by the complete actor.
- Train for 10,000 updates, batch 16, AdamW at 1e-4, betas (0.9,0.95), weight
  decay 0.01, global gradient clip 1.0. Linear warmup 500 updates, then cosine
  decay to 1e-5. Use FP32 trainable weights/moments, writer BF16 autocast,
  deterministic math attention and EMA decay 0.99. Seeds: model 20266420,
  CPU image sampling 20266421, CUDA flow noise/time 20266422.
- Standard conditional flow loss: U~N(0,I), t~Uniform(0,1),
  Y_t=(1-t)U+tY, target velocity Y-U. No foreground weighting, raster repairs,
  character selection or rejection sampling. Evaluation uses the final EMA,
  16 explicit-midpoint steps and a zero binary threshold.
- Save exact optimizer, model/EMA, sampling/noise/global RNG states and sample
  journals every 250 updates and on a handled interruption. Retain current and
  immediately previous generated checkpoints. Before continuing beyond update
  three, restore the update-two checkpoint and replay update three; require
  identical writer/EMA/optimizer/RNG, selected images and loss. Count the extra
  replay in actual compute, with no duplicated journal entry.
- One owned job on physical GPU 0; no shared GPU-1 service. Require idle GPU 0
  at startup, at least 24 GiB available host RAM, swap below 75%, less than
  22 GiB allocated GPU memory, and a 24-hour stage ceiling. Save and interrupt
  cleanly if host resources cross their limits. Preserve any failed attempt.

## Fixed component evaluation after training

Use 128 complete development-answer blocks: 64 source examples per language,
uniformly without replacement using CPU seed 20266430, followed by one uniform
block per selected example. Pin the development cache report SHA
`14b088cff0c91175570f4635af7363419225cb36c8784d11efa679b0993bb369`.
Do not inspect their generated outputs before the final EMA is fixed.

Generate once from correctly paired visual features and once from a one-case
cyclic permutation of the features, retaining targets and initial noise per
case. Report raw foreground F1, exact raster match, blank fraction, per-case
arrays and first eight fixed raw outputs, with no repair or reselection.
For qualification of this writing component, require mean foreground F1 at
least 0.90 and a paired mean advantage over shuffled conditioning at least
0.20. These thresholds say nothing about answering questions. No sealed data
is opened. Same-font development records may share forms with training; no
unseen-script or unseen-font claim is allowed.

## Language stage remains required

If the writer qualifies, initialize the complete actor's writer from that EMA
and train on full observed Chinese/English prompts and answer-image histories.
Register its actual optimizer, budget, semantic supervision, stopping/decoding,
autonomous answer metrics and question controls before its quality audit.
Use the measured source context with its vocabulary removed. Offline teacher
assistance remains permitted and attributed. A failed writer stage requires a
recorded repair, not a language claim based on visual reconstruction.
