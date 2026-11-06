# Native-context Chinese image-language training

## Question and scope

The fresh image writer failed complete answers even on training data. In the
source-reader intervention, retaining only the final task-frame KV lost 17 of
19 correct training answers. This run retains the full frozen native visual
context and trains continuous image-history input, image output and structural
stopping. The inherited externally pretrained body remains part of the model;
this is neither training an 8B language model from scratch nor a lightweight
independent language result.

Use all 3,193 admitted Chinese CMRC training records from the existing 12,976
record cache, with authenticated original prompt pixels and complete image-code
targets. English records are excluded from optimization. No development answers,
source vocabulary projection, tokenizer or teacher service enters optimization
or deployment. Physical raster preparation and offline scoring remain separate.

## Fixed model and optimization

Use the completed V79r2 architecture without changes: 36 frozen inherited native
layers, full prompt KV, 46,899,713 trainable image-interface parameters, qualified
fixed codec, clean endpoint prediction with uniform time and the registered
32-step sampler, learned stop at 0.5, 128 block cap. Retain all original image
target blocks, including internal whitespace; remove only the two structural
white termination blocks already removed in the verified code cache.

Seed 20268000 initializes new parameters. Train 40,000 updates, batch four,
AdamW (0.9,0.95), weight decay 0.01, gradient norm clip one. Peak learning rate
1e-4, 500 update linear warm-up, cosine to 1e-5. The original per-example equal
weight clean and balanced stop loss is unchanged. Use up to 16 image blocks per
example; Chinese targets have at most seven. Per-layer non-reentrant gradient
checkpointing recomputes only new image-history queries through frozen layers.

Full native KV is too large for an unbounded disk cache. A shuffled exact-epoch
sampler selects 32 prompt records per replay pool; each is read from actual
pixels once, checked against the existing 52-state BF16 memory digest, then
retained only in CPU RAM. A separately serialized shuffled sampler draws batch
four for 128 updates per pool. Refresh sequentially, never concurrently. This
gives 160,000 optimization draws and 10,016 prompt prefills over 313 pools.
The last pool contributes 64 updates. Report actual timings, bytes, unique
records and draw counts; pool reuse is an efficiency tradeoff, not IID sampling.

Right-pad heterogeneous KV batches and explicitly exclude padded keys. Keep
each source's actual rotary offset. The full prefix stays immutable. Target
images supply causal history only after their corresponding writing decision.

## Execution prerequisite and persistence

Require the authenticated supplemental V79r2 execution receipt. On four actual
Chinese training prompts spanning short and long targets, exercise pixel
prefill, BF16 memory identity, heterogeneous prefix batching, three complete
updates and exact third-update replay after restoring model, Adam, both sampler
states and all noise/RNG states. Verify all trainable gradient groups become
finite and nonzero after initialization. Retain this preflight separately;
discard its new parameters before production initialization. No small random
network substitutes for the actual inherited body.

Checkpoint only new interface weights, optimizer, record/pool cursors, RNG and
source hashes. Recompute the bounded active pool from actual pixels on resume;
restore optimization RNG after that computation. Shared frozen weights are
referenced rather than duplicated. Preserve current and immediately previous
checkpoints. One project GPU0 lock covers the entire training process.

## Outcome

Retain unmodified autonomous images on four owning training prompts after
updates 1,000, 5,000, 10,000, 20,000 and 40,000. These are diagnostics, not
generalization evidence or early-stopping selections. Do not select a checkpoint
by these outputs. After the declared final update, run the original full 192
question / 384 generation audit with question-erased controls. Original Chinese
gates remain exact >=0.5, foreground F1 >=0.8, both-correct pairs >=0.35,
question gain >=0.1, positive lower 95% gain bound and learned stopping >=0.9.
Report English separately as untrained transfer. Independently score retained
physical pixels. Failed outputs and failed execution prerequisites remain
visible. Neither a completed optimization budget nor a training-set fit
establishes the general ILM goal.
