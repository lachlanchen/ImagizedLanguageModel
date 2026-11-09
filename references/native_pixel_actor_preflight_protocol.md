# Full-native PNG-to-pixel actor: execution and learning preflight

The previous goal turn completed real optimization and a reviewed publication
update. The short/long cached-state tests are now finished: the long head fits
512 training first blocks exactly but no development first blocks. They do
not qualify a useful ILM. The next step must exercise actual prompt PNGs,
complete image histories, direct pixel output and learned structural stopping.

## Interface choice and initialization

Preserve the later full-native prompt-KV interface. The source-context audit
already rejects masking it to 52 positions. Do not revert to the earlier V68
custom two-query geometry merely to reuse its fitted condition projection.
Use the unchanged source observer and all 36 contextual layers of V80/V81.
The source observer accepts pixels and fixed numeric structural vectors, with
no tokenizer, lexical embedding table, language output head or teacher service
in deployed inference. External foundation pretraining remains substantial.

The single native writing state has width 4096. Construct the direct pixel
density at that width, using per-state parameter-free LayerNorm and zero/one
coordinate normalization buffers. Reset its condition projection. Transfer
only the other physical decoder tensors from the completed 16,000-update
head: physical positions, signed-bit projection, causal layers, start vector
and normalized pixel-pattern output. Its previous 512/512 fitting result is
not the performance of this new interface.

Reuse the completed V80 image-history encoder, input scale and stop weights;
discard its output-code prediction flow. Add the existing V81 learned start
query and rank-8 suffix q/v/o adapters. Their scope includes backward
checkpoint recomputation but excludes prompt prefill. Consequently the
source prompt KV must remain bit-identical after real suffix learning.
This composite new candidate does not isolate a change in loss alone.

The frozen qualified image codec is retained only to encode actual observed
history pixels. Its decoder must never generate an output. Every written
block comes directly from the normalized causal 2-by-4 physical-pattern head.
Past-image codes are continuous physical features, not word identities.
The retained codec's unused decoder weights count in complete model size.

## Complete supervised objective and autonomous generation

Teacher training reads complete answer images and conditions query t on the
prompt and completed blocks before t. Use the mean physical-pattern NLL plus
the mean of balanced continue/end softplus losses, each with weight one.
This disclosed weighting is not the unscaled full joint likelihood. Current
and future target pieces cannot condition their own predictions. Generation
receives only prompt pixels, rereads its own emitted images through the frozen
encoder, uses the learned stop probability threshold 0.5 and has a uniform
16-block cap for this preflight. Reaching the cap without stopping is failure.

Select cases by source order and complete reference length, before any native
model call: first CMRC and Dolly training records in the previous 512-case
cohort with 2--8 complete answer blocks, the first CMRC development article
pair, and first Dolly development case with 2--8 complete blocks. The actual
selection is training indices 0/262 and development indices 512/513/640.
No answer is truncated. All five prompt PNGs must decode to exactly the
original owning binary pixels. References are separate offline assessment
artifacts and never arguments to autonomous generation.

## Execution checks and optimizer replay

Freeze and bind code, data, the completed native checkpoint (SHA-256
2e267b023118552e2462971c5f8e0dc49b57f4e401ef7ba58082c05fc85fc0f5),
numeric controls, qualified codec and physical decoder checkpoint before GPU
execution. Use the existing PyTorch 2.3 runtime and one GPU0 project lock.
Require 24 GiB host RAM available, at most 75% swap and 20 GiB GPU0 free
before loading the real multi-billion-parameter body. Preserve GPU1 and
foreign services. Bound peak allocation to 22 GiB and execution to 1,200 s.
No Office or competing project model job may overlap this execution.

Verify future-image exclusion by changing the second target block while
requiring the first two query states unchanged exactly, before and after
learning. Compare complete-history and incremental contextual states with
the existing normwise 1e-4 criterion; retain actual absolute/relative errors
and emitted-pixel agreement separately. Repeat the pixel decoder's shifted
target causality and cached/full-prefix checks at all 256 physical positions.
Record full native KV retention and actual generated-image feedback hashes.

Execute one update on the complete selected CMRC answer and one on the
complete selected Dolly answer, at rate 1e-5, AdamW betas 0.9/0.95, decay
0.01, foreach false, clipping at one. Save after update one; execute update
two, restore update one and replay update two. Model and optimizer state,
loss and gradient norm must match exactly. There are two retained logical
updates and three actual optimizer calls. Check finite nonzero gradients in
the physical head, history encoder, start query, suffix adapters and stop
head. Frozen source parameter identities/versions must remain unchanged.

After the two updates, recompute all five actual PNG prefills and require
their initial states and complete KV hashes unchanged. Generate all five
full answer streams with learned stopping, then repeat the first stream for
bit-exact deterministic replay. Also run one separately labelled forced
two-block path control to guarantee actual feedback execution if the learned
stop emits nothing. This forced control is never scored as an autonomous
answer. Independently score complete emitted strips and stopping, retaining
every original PNG. Two optimizer updates are an execution/gradient check,
not language training or quality qualification. No semantic or historical-
script capability follows. Preserve failures without automatic retry.

The offline CPU auditor verifies every complete output PNG against actual
emitted tensors, scores whole strips with white padding for unequal lengths,
and requires correct length, learned stop and no cap for exact credit. Empty
outputs may have an explicitly labelled white display placeholder; that
placeholder never counts as generated content. Instrumentation records every
actual encoder input/output without changing arguments or return values.
The audit checks those inputs against emitted blocks and re-encodes them on
CPU against saved GPU codes at fixed absolute tolerance 1e-4. It does not
independently replay the entire inherited reader or all optimizer operations.
