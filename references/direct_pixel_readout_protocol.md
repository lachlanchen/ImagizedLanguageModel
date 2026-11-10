# Direct pixel loss on the same genuine reader states

This is a conditional follow-up, not a result or a replacement for the running
genuine-copy experiment. Prepare it only after that experiment's qualification
is terminal and a physical criterion fails. If its only failure is initial
stopping, decline this ablation: changing the drawing loss does not address
that failure. If the parent passes, assess its complete autonomous answers first.
No new neural stage runs while the current cache or supervisor owns the GPU.

The current writer minimizes an anchor-vector discrepancy as well as physical
pixel loss. Its rank-2048 output cannot exactly represent every 4096-dimensional
input anchor. That particular regression floor need not bound image writing:
the final output is a1024-pixel glyph, and the image-feedback encoder is a
separate function. Test the direct conditional image objective

\[
q_{\psi,\phi}(I\mid h)=\operatorname{Bernoulli}
  \bigl(\sigma[D_\phi(G_\psi(h))]\bigr),\qquad
\mathcal L=\operatorname{BCE}_{\rm balanced}(D_\phi(G_\psi(h)),I).
\]

Use the existing foreground/background-balanced BCE. Do not regress the motor
vector to the source input embedding and do not add cosine or contrastive
anchor losses. This removes an auxiliary coordinate constraint, not all model
capacity limits. Pixel-wise conditional losses can still average uncertain
forms. Clear held-context glyphs and autonomous language remain required.

Two fixed arms restart from the preserved original language checkpoint:

1. `g_only`: train G with pixel loss; keep the entire D fixed. Relative to the
   parent, this isolates removing semantic MSE and cosine terms.
2. `g_and_decoder_input`: use that same pixel objective and additionally train
   D's layer-normalization/two-linear input block. Keep D's convolutional
   renderer fixed. This permits a different internal motor coordinate system;
   it changes trained capacity as well as the optimized function, so it is not
   a loss-only comparison.

Both arms retain the original architecture and total parameter count. E,
semantic affine buffers, stopping weights and the pretrained body remain exact.
Each actual emitted image is still encoded by the unchanged E for continuation.
The drawing vector need no longer claim to be a source semantic embedding.
Neither arm uses a runtime font, glyph table, lexical head or teacher.

Reuse the completed8,172genuine-copy training states and3,855original nonterminal
gold-image-history states. Do not recache the body, use held wording states in
fit, or read C3 inputs/references. Use the parent's seed202611111,18,000batch128
AdamW updates at2e-4, weight decay0.01, clip1 and the same cosine schedule.
Sample half each batch from each training pool. This matches the prescribed
sampler and update budget; parent sampler draws were not separately retained,
so bitwise equality to its random index sequence is not claimed.

Qualify each arm on every training-copy, held-wording and original-history
state with the parent's unchanged physical thresholds. Retain all emitted
pixels and per-state metrics. Cold-replay every baseline bitmap against the
parent's saved baseline to validate comparability. Inspect all three fixed
64-triplet boards before any autonomous assessment. Existing histories measure
retention, and held wording uses known identities. No additional generalization
claim follows from either.

Only a passing, actually reviewed arm may run the original forty-stream
autonomous assessment with natural stopping and64glyph caps. Review all ten
raw boards before reference scoring. C3 remains excluded from training and
model selection; any later registration must preserve its prospective data
lineage and freeze the candidate before its first C3 prediction.

Each fit/qualification is a separate, serial CUDA0process under the same project
mutex. Use the existing48GiBentry/32GiBruntime RAM,8GiBVRAM,1800second stage and
300second pressure-wait guards. Autonomous assessment uses the unchanged
1200second native-reader admission. Do not alter any completed source binding,
weaken a gate or stop foreign work. Record both failures and successes.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/train_direct_pixel_readout.py --stage prepare
# After preparation, run each arm's fit and qualify sequentially on CUDA0.
# Assessment additionally requires a passing gate and actual image-review receipt.
```

No direct-pixel fit, qualified glyph improvement, complete answer improvement,
efficiency advantage or new foundation model is established by this protocol.
