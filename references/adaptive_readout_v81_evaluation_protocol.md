# Evaluate the completed readout checkpoints through a common incremental path

Prospective evaluation amendment, 7 November 2026. Both matched arms finished
2,000 optimization updates. The original evaluation stopped at adapted case
00: autonomous and full-history first-block pixels agree exactly, but their
stop probabilities are 1.029378449146634e-8 and 1.0293823571316807e-8. Requiring
those floating-point values to be bit-identical rejects a difference of
3.908e-14. The original failure and partial outputs remain unchanged.

This evaluation performs no optimization. It fixes the two completed
checkpoints, original 32 training questions, case order, seeds, 32 flow steps,
0.5 stop threshold, 128-block cap, targets and original training-fit criterion.
It does not complete or relabel the failed execution. Autonomous generation
uses the unchanged model method; no target information enters that call.

For the correct-history/known-length control, compute the initial state using
the same incremental start as autonomous generation: the native initial
state for the control, or the learned query through `start_prefix` for the
adapted model. Advance the cache with the supplied preceding target codes,
two positions per block. Predict every reference block and record the final
stop probability. Require the common first pixels and stop probability to be
exactly identical to autonomous generation. This is an execution identity,
not a quality criterion or a test of free generation.

Retain the original batched full-history control as a separate diagnostic,
including its states, coordinates, pixels and stop probabilities. Report its
discrepancies with the incremental control rather than silently discarding
it. The full-history hidden states remain subject to the existing V81r2
normwise numerical bound; record the original absolute 1e-4 outcome too.
No pixel or stop threshold is relaxed. The adapted autonomous case 00 must
replay the failed execution's saved output exactly; all 32 control autonomous
outputs must replay the completed control outputs exactly. No best checkpoint
or output selection is permitted.

Save all 64 autonomous answers and both types of correct-history output.
Independently rebuild owning prompts and targets, check retained raw-image
and PNG equality, replay both control-coordinate decodes on CPU, and score
every answer with the independent physical scorer. Supplied-code codec
reconstructions remain a separate upper-bound control. Report full case
results and all original thresholds: adapted free exact >=24/32, mean free
foreground F1 >=0.98, and both exceed the matched control. These are training
fit thresholds only. No general language or lightweight-from-scratch claim
follows even from a pass.

The original run directory is read-only. New results go to
`artifacts/adaptive_readout_v81eval_20261107`. Only one GPU0 model process may
run, followed by an explicitly CPU-only verifier. Implementation and this
protocol must be committed before execution. Public receipts identify both
the failed execution and this evaluation separately.
