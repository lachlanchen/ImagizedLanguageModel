# Qualify complete native-context image continuation before training

Run on the actual authenticated inherited8B contextual body and qualified fixed
writing codec, not a reduced random model. The new implementation has a learned
image-code input adapter, clean-code output density and learned structural stop.
Use the unchanged V78 first-case full prompt KV as an execution fixture, the
V77 exact initial BF16 state, and the original source rotary offset. Preserve
the full visual/prefix context. No compact-state sufficiency is assumed.

This is an execution/gradient check only. It performs no optimizer updates and
makes no language-quality or training/generalization claim. The fixed copy
case stays evaluation-only. Gold image codes used in assisted causal/gradient
checks are never fed to autonomous generation. Commit exact source and this
protocol before execution; freeze them after measurement.

On physicalGPU0 with seed20267910, four CPU threads, deterministic math/eager
attention and no TF32, check all of the following:

1. Feed the original source first-choice numerical input embedding through the
new functional full-prefix continuation. Its no-gradient and checkpointed
forward states must agree with the independently retained V78 unmasked next
state within1e-5. The source token-row lookup is diagnostic-only and never
belongs to the deployed image model.
2. Backpropagate a fixed numeric linear functional of that native next state
to the input embedding through all inherited layers. Its gradient must be
finite and nonzero. All inherited parameters remain frozen, without gradients
or version changes. The original prefix KV must remain byte-identical.
3. Add17 zero prefix-padding positions while retaining original valid lengths
and rotary coordinates. The same one-query hidden state must agree within
1e-4. This checks padding visibility separately from positional encoding.
4. Encode the original first copy reference into actual frozen writing codes.
Compare teacher-forced parallel conditions with repeated two-tile native
advances using the same supplied images. Maximum state difference must be at
most1e-4. Replacing future image codes must leave the first two conditions
unchanged within1e-5, while later states change. These assisted checks receive
no autonomous-answer credit.
5. Compute the complete supervised flow-plus-stop loss on those actual codes
and backpropagate without taking an optimizer step. New-head gradients must
be finite and nonzero. A separate contextual-state functional must yield
finite nonzero image-input-adapter gradients. Initial zero clean-output
projection can block an image-adapter gradient from the first flow loss;
this is recorded rather than treated as evidence of learned understanding.
6. Round-trip only the image adapter/clean flow/stop checkpoint and verify
exact new-parameter equality; do not save another copy of source weights.
Generate an unforced two-block-capped output, then re-encode its actual
emitted pixels and check every feedback digest. Separately force the stop
bias positive, exercise the empty stopped branch, and restore the original
bias exactly. Neither untrained output receives quality credit.

Retain numerical vectors, actual gradients, raw physical output tensors/PNGs,
source/cache hashes, causal and incremental comparison summaries and peak
allocated GPU memory. Every inherited layer is exercised in forward and
input-gradient propagation. No source vocabulary module, output projection,
teacher service or fixed glyph inventory is attached to the image model.
Require peak allocation below the physical24GiB card capacity. Independent
CPU checks authenticate source fixtures, compare retained numeric references
and gradients, verify parameter/checkpoint hashes, and replay all feedback
encodings using the fixed codec. They do not replay the entire8B backward pass.
Release the owned model/job after the check and keep the full ILM goal active.
