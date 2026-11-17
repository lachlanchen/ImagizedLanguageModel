# Matched 64-dimensional continuous and 64-bit visual-plan codecs

This comparison qualifies a proposed global writing plan, not language. The
new plan must control a faithful neural drawing function and reduce actual
physical refinement before conditional prior/correction learning is registered.
It follows the failed whole-form auxiliary trial; it does not repeat the old
768-dimensional codec, select an old writer snapshot or repair frozen readings.

## Sources and one changed computation

Use all 68,948 original prepared training-support cells and the separately
registered exact-mask-hash partition. Repeated physical images share a partition;
the empty mask trains. Keep every source byte and its multiplicity. Physical
validation withholds masks, not books, semantics, styles or language groups.
No DRCD confirmation, development answers, teacher labels or old judge is used.

Both fresh codecs have 2,030,465 parameters, the same 32/64/96/128 convolutional
channels, 64-dimensional global bottleneck, initial tensor bytes (CPU seed
202611195), decoder and source order. Control draws from tanh-bounded continuous
values. Candidate draws from their hard sign, using the implemented straight-
through reconstruction gradient. Only this quantization changes. No pretrained
weights, font, image bank or per-form output embedding is loaded.

For each complete image use boundary-weighted BCE, .5 replicate-padded Sobel L1
and .5 foreground Dice. Foreground is one; normalize the weighted BCE within
each image, then average whole images. This is a reconstruction objective, not
an exact image likelihood. Both modes use the same loss and precision.

## Real support preflight and fixed fit

Before each arm's fit, accumulate finite gradients over **every complete training
image** in microbatches of 128. Require finite nonzero encoder-bottleneck and
decoder gradients. Make one AdamW update at .0002 with weight decay .01 and
norm clipping one, measure its complete training objective and record changed
tensors. Reset every trial tensor exactly and discard that optimizer. The
preflight tests update execution, not an assumed one-step quality gain. Abort
without a fit on missing gradients, nonfinite values, failed reset or resources.

Then fit each mode serially for exactly 6,000 updates with batch 128. Use one
fixed training-index permutation (CPU seed 202611196), cyclic indexing without
dropped images, and identical batches in both arms. AdamW betas are (.9,.95),
weight decay .01, gradient clip one. Linearly warm up for 250 updates to .0002,
then cosine decay to .00002. Use FP32, TF32 off and deterministic cuDNN, no
latent noise, EMA, class balancing or schedule selection.

At update three, save the actual model and optimizer, execute the update, restore
and replay that same batch once. Require exact loss, gradient norm, model tensors
and optimizer state. Retain only the replay-equivalent scheduled update. Record
the extra 128-image trial per arm and both complete-support preflights as costs,
apart from the 768,000 retained fitting draws per arm. Save only the prescribed
final checkpoints for qualification; logs do not authorize intermediate selection.

## Complete physical evaluation and fixed conditions

Initially and finally decode **all** source images in their original order.
For the binary arm call the deployed hard encoder and bit decoder. Preserve
every output pixel, its 64-bit plan, XOR correction event count and all continuous
control values. Check exact XOR recovery without treating that identity as a
quality result. Independently recount metrics from raw arrays on CPU. Empty-
empty image F1 is one in macro-form averages; pooled foreground F1 uses summed
TP/FP/FN. Macro metrics give each distinct physical mask equal weight and retain
all occurrences. Continuous vectors have no claimed eight-event transmission.

Qualification requires all of:

- Binary pooled foreground F1 at least .98 in training and physical validation.
- Binary validation macro-form F1 at least .98 and macro exact rate at least .80.
- Binary validation pooled F1 no more than .005 below matched continuous control.
- Continuous validation pooled F1 at least .98, to qualify the common codec ceiling.
- Zero binary false blanks and false ink over the complete source support.
- Binary mean of eight plan events plus actual XOR-tree events no greater than
  .60 of the previously measured 97.30248883216336 direct-tree source events.

Event counts are not bits, throughput or language efficiency. This is a fixed
finite-budget recipe; failure is not an impossibility theorem for binary codes.
If fidelity or event reduction fails, retain both final models, all images and
the failure; do not extend unchanged epochs, tune thresholds or select snapshots.
If all conditions pass, only a separately registered conditional image-generation
comparison may follow. No language fit, goal completion or parity claim follows.

## Resources and review

Use one GPU0 owner under the project mutex, preserving foreign services. Entry:
48 GiB available host, 20 GiB free GPU, 48 GiB disk and low memory pressure.
Runtime: 24 GiB host reserve, 8 GiB allocated GPU and 8 GiB process RSS, two CPU
threads, 2,400 seconds including at most 300 seconds pressure wait; external
timeout 2,460 seconds plus 30 seconds termination grace. Bind packages, code,
sources and partition before any neural call. Preserve failed executions; no
automatic retry. Record both arms and all preparation/evaluation costs. No
contextual foundation is loaded, and no cost is claimed for a whole ILM.

Export fixed source-index sheets: the first 64 distinct held masks and first
16 distinct training masks, including empty, in original source order. Preserve
references and actual outputs with nearest-neighbour enlargement only. These
are physical reconstructions, not answers, OCR readings or a language benchmark.
