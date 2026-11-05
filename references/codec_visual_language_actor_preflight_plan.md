# Complete semantic reader with an independent continuous writing space

5 November 2026. V66's direct reconstruction ends at 56.563% foreground F1,
failing its unchanged component criterion. Its fitting success cannot stand in
for broader fidelity. V67 audits the existing fixed V34 output codec on the
same 128 development blocks; its source/latent/image verification must pass
before this new complete-model preflight. The failed V66 weights are not used.

## Model and rationale

Use `CodecVisualLanguageActor` with the actual complete inherited 27-layer
visual reader and 36-layer contextual body, retaining the established FP32
compute/BF16 storage repair. Attach the verified frozen `CodecBlockWriter`.
Read prompt/history images in the input encoder's 4096-dimensional space and
predict output images in the codec's separate normalized 768-dimensional space.
This avoids learning an inverse of the semantic input encoder. The output
codec was trained locally in V34; semantic foundation pretraining is external.

Train rank-eight query/value/output adapters, two structural writing queries,
a shared 4096-to-768 continuous projection and the learned stop classifier.
No character inventory, source vocabulary, OCR or external model enters the
actor. The projected field is layer-normalized to the codec's native scale.
All codec parameters stay frozen; its encoder supplies detached labels only.

```text
h_b = body(input_encoder(prompt), input_encoder(completed_blocks), queries)
z_b = LayerNorm(output_projection(h_b))
target_b = stop_gradient(output_codec.encode(actual_target_block))
L = mean ||z_b-target_b||^2
  + mean ||output_codec.decode(z_b)-actual_target_block||^2
  + balanced_stop_loss.
```

Each physical block has two 32-by-32 tiles, with no character segmentation.
The codec decoder receives only predicted continuous vectors during generation.
The next history item is the resulting thresholded image, not an invisible
latent proposal. Stopping is learned separately from rectangular white padding.
Teacher forcing independently encodes each actual block and masks current/
future pixels from its query. Uniformly sample at most 16 block losses while
retaining complete observed history and all actual answer blocks.

The earlier direct-4096 bridge prototype depended on V66 qualification and
was not GPU-tested. Its source/CPU checks were retained privately. This output
space replaces that prototype, without changing any V64–V67 source or score.
Earlier V35/V47 codec-language failures remain relevant: visual codes and
regression do not guarantee language. The present larger inherited semantic
body is a different candidate whose actual language behavior is unverified.

## Required prerequisites and hardware checks

Require V67's output-interface gates and independent original-image/latent/
raw-raster verification. Pin the exact V34 checkpoint, V67 report and verifier
receipt before execution. Verify V66 is terminal and GPU 0 idle. Keep one owned
GPU-0 process with available host RAM >=24 GiB, swap <75% and allocated CUDA
memory <22 GiB. Do not touch GPU 1 services or introduce a GUI/reader engine.

Use actual complete training examples, including the largest established
prompt and context rows. Check full backward with real Adam moments. Frozen
base/encoder/codec parameters must have no gradients, while finite gradients
reach the output projection, queries, stop head and first/last adapter outputs.
Report exact counts and measured memory/time rather than a size estimate.
The codec's separately tested input gradients must remain available while its
parameters stay frozen. Predictions and losses must stay finite.

For same-shape interventions on future answer pixels, compare earlier raw
query states, predicted visual states and stop logits, not total supervised
loss. Label losses may change. Confirm independent-encoder isolation. Compare
cached/full prefixes under the existing relative tolerance .02 and report
absolute differences. Restore/replay an actual optimizer update exactly, with
all trainable tensors, optimizer, sampling/global RNG and outputs. Test
trainable state round trips and deterministic raw-image generation. Preserve
any failure; untrained output is execution evidence, not language quality.

## After execution

Only after the full-model preflight passes, register actual instruction
learning and its final autonomous question/answer audit. It must use complete
Chinese/English prompts, image-only feedback, learned stopping and question
interventions; no reconstruction or supplied-answer ranking can replace it.
Then maintain the Nature manuscript with the verified V65/V66/V67 distinction,
matching editable figures, source-exact raster examples and final evidence.
The complete independent image-language goal stays open throughout.
