# Joint exact raster language V55: frozen matched experiment

Date: 2026-11-05. Written before implementation or optimization.

## Question

Can the same exact raster density learn better conditional language when its
reader is trained jointly under normalized image likelihood? V54 learned a
copy-capable image density with a frozen reader. A post-result intervention
found a 5.411-nat whole-context effect, but autonomous images still lost their
intended form. Before redesigning the writer or increasing scale, test the
ordinary end-to-end learning path that V54 deliberately held fixed.

The network, binary observation space, exact pixel factorization, spatial
masks, copying path, rendering and deployment contract remain unchanged.
There is no new learned vocabulary, teacher, codebook, OCR, image bank, image
selector, loss mixture or decoding repair. This is an intervention on gradient
flow, not a claim to have invented autoregressive image modeling.

For reader parameters phi and density parameters theta, optimize

    L(phi, theta) = E[-log p_theta(X | h_phi(H), a_phi(H))].

Both hidden state and aligned image field receive the score gradient. V54
fixed phi; the new joint arm permits it to adapt to the same normalized joint
image probability. The objective can still fail through insufficient data,
capacity, global image structure or short-context dependence. Joint training
does not guarantee success.

## Matched arms and fixed optimization

- Both arms start from V54 EMA, checkpoint SHA-256
  `f2c3fab74981541243a963c8cffab6bbbd8c7f9d051512b228347db04a57a9ee`.
  Reinitialize the optimizer and EMA from those identical weights.
- Frozen control: update the same 2,434,592 density parameters only.
- Joint candidate: additionally update all reader parameters that participate
  in the likelihood. The unused scalar contrastive temperature is frozen in
  both arms. Runtime remains 18,711,841 parameters.
- Keep reader dropout disabled in both arms, as in V54. Training mode alone
  must not accidentally add a second intervention. Reader gradients are
  enabled in the joint arm despite its evaluation-mode dropout.
- 8,000 further updates per arm; batch eight streams; 16 stratified causal
  positions per stream, including the last. Same data and RNG draws in both
  arms; training seed 20265510, dataset seed 20265511.
- New deterministic train-stream samples from the unchanged public Chinese
  manifest, SHA-256
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`.
  Source partitioning and fixed 32px/26px canonical renderer are unchanged.
- 25% supplied-image motor exercises with zero hidden state, as before. The
  other rows use predicted conditions. Copy rows cannot train the reader by
  leaking their true target back into its history.
- Pure binary image NLL, divided by 1,024 for optimization; no auxiliary loss.
- AdamW: density LR 2e-4, reader LR 2e-5; betas (.9,.95), weight decay .01,
  400 warmup updates, cosine decay to .1 peak, joint global norm clip one,
  BF16 and EMA .999. No learning-rate or sampling sweep.
- One physical RTX 4090 (GPU 0), sequential arms, peak allocated VRAM <18 GiB.
  No simultaneous heavy project job. Complete both arms before opening their
  development results. Stop only for an actual numerical/resource failure.
- Checkpoint full optimizer, model/EMA, RNGs, deterministic data offset,
  source/protocol, corpus and initialization hashes. Exact-resume smoke and
  reader-gradient isolation must pass before either production arm starts.

## Fixed measurements and decision

Use the same deterministic development windows and audit seed 20265420 as V54:
512 exact NLL targets; 256 first samples, greedy outputs and supplied-image
motor controls; 64 four-step visible feedback streams. Generate before creating
the 9,932-style evaluator bank, with every target covered and pixel deduplication.
No output lookup or replacement. Retain all raw outputs, per-example NLL,
identity and fixed galleries. Source-record bootstrap uses 2,000 draws.

Apply V54's eight unchanged qualification gates to each arm. Report candidate
minus matched-control paired NLL and identity intervals, not just comparison
with the shorter V54 training run. A broader language-training step requires
the joint candidate to pass all eight gates and improve ordered target NLL
over the matched frozen control with a strictly positive paired 95% lower
bound. If those fail, report the result as partial/rejected, preserve any
working components, and do not claim high-quality ILM, language-model parity,
novel priority or general efficiency. Sealed records remain unopened.
