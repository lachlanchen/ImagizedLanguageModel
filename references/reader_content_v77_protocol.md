# Locate content loss before another image-language training run

The completed full-corpus image writer fails new copying and fails its earlier
32 training questions even with correct answer history. A matched head-only
time-law repair improves assisted near-clean reconstruction without repairing
free answers. This diagnostic asks whether the imported reader can recover
content from those exact prompt images. It performs no optimization and does
not qualify a useful image-output language model.

## Fixed cases and helper boundary

Use all 12 previously prepared V75 copy requests, in their original order,
and exactly the 32 CMRC training questions fixed by the V74 receipt and used
in the V75 actual-image diagnosis. Reconstruct the same folded physical image
inputs and verify their digests against the original actor calls. Authenticate
the original training text source, and re-render every prompt and canonical
reference using the unchanged complete-line renderer to confirm its identity
with the original physical strips. Text controls contain the owning source
prompt, never a separately appended answer. All sources are already local.

For each of the 44 cases generate a normal-image helper response and a matched
source-text helper response. For all 12 copy requests additionally generate a
fully blanked prompt-image response at the same physical geometry. This gives
100 fixed helper generations. Use exactly the existing native-budget observer
(factor 32, area bounds, RGB Pillow bilinear then L) and the same authenticated
before/after numeric task controls. Retain source/transformed PNGs and hashes.
The blank route removes all prompt ink; it is not a question-only intervention.

Reuse the frozen imported image/contextual body with all adapter outputs zero,
and remove its unused writing decoder. Its vocabulary embedding module remains
absent. A separate offline evaluator may use the authenticated source tokenizer,
source token-row lookup and temporary output projection for diagnostic greedy
text decoding, as in the existing V71 helper. These must not be attached to or
exported with the independent ILM runtime. No model service, download, teacher
training label, output repair or new deployed language head is introduced.

## Generation and interpretation

Use deterministic physical GPU 0, four CPU threads, seed 20267710, no TF32,
deterministic cuDNN and math/eager attention. Decode greedily with the same
64-source-token cap and authenticated end IDs as V71. The temporary source
projection is stored in BF16 and contracted in FP32. Preserve every selected
ID, selected logit, corresponding hidden state, unmodified text and stopping
decision. Retain the complete 52-state suffix at normal-image prefill, and
require its BF16 bytes to equal the original V75 copy or training-reader memory.
Round the first projection state to BF16 for every route. On normal-image
calls this makes the first source projection use exactly the compact memory's
last stored vector, rather than additional FP32 precision from prefill. Later
states use the original contextual continuation and full KV history.

This equality binds the shared observation to the failed ILM call. The text
helper nevertheless continues with the full layerwise visual/context KV cache,
whereas the ILM writer sees only 52 final-layer states. A correct full helper
response therefore establishes capability of the imported reader with that
cache, not sufficiency of the compact memory or learning by the ILM. Report
the first decoded source piece's nonempty trimmed match to the canonical
answer prefix separately: this first projection uses a final state present in
the compact memory. Also report whether that prefix excludes the paired
answer. Prefix agreement remains a limited string diagnostic, not complete
language understanding or an independent image-output result.

Report stopped literal canonical-answer exactness after outer whitespace trim,
character edit accuracy, owning/paired reference inclusion, first-piece prefix
agreement, stopping and truncation, separately for copy normal/text/blank and
training normal/text routes. All outcomes remain descriptive; no new language
gate replaces the failed original audit. Do not alter old sources, protocols,
checkpoints or scores, and do not open sealed examples.

## Verification and next decision

Before execution, commit and push this protocol and exact implementation.
Preparation and verification are CPU-only. Use the shared project GPU lock for
the single GPU stage, record its owned PID, and release the model on exit.
Independent CPU verification reconstructs the original images and declared
resize, checks source text/render identity, compiles numeric controls again,
authenticates every retained output, decodes all IDs, recomputes scores and
replays all temporary projection choices within the existing V71 tolerances
(selected logit error <=1e-3, regret <=1e-4). Exact argmax agreement and near ties
are reported separately. It does not replay the complete vision/context body.

If source-text controls work but image calls fail, investigate visual import,
geometry and the image interface before training another writer. If image
responses work while the ILM outputs fail, inspect the compact-memory boundary
and conditional readout; the helper's full KV advantage must remain explicit.
If first-piece projections work from identical compact states, they identify
some available answer information before the image writer. None of these
outcomes alone supplies a successful independent ILM or an efficiency claim.
