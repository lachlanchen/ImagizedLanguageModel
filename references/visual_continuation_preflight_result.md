# Full visual-continuation adaptation fits on one 4090

The registered real-body preflight passes. It trains the query, all 36 new
continuation residuals and the image readout/stopping heads through complete
answer computations, with frozen native prompt memory. Peak allocated VRAM is
18.123 GiB. This establishes the measured implementation's feasibility; language
quality has not been evaluated and a full fit is not yet registered.

The [protocol](visual_continuation_preflight_protocol.md),
[registration](evidence/visual_continuation_preflight_registration_20261112.json),
[actual GPU execution report](evidence/visual_continuation_preflight_20261112/execution_report.json)
and [independent CPU audit](evidence/visual_continuation_preflight_20261112/audit.json)
retain the tested candidate, cases, gradients, optimizer evidence and limits.

## Complete training cases and checks

The cases are deterministic extrema of eligible authentic training sequences.
The old feedback-ceiling diagnostic controls and their source groups, and the
current held-out QA groups, are excluded. Only training data enter the updates.

| Case | Visible prompt glyphs | Complete target glyphs |
| --- | ---: | ---: |
| Shortest QA | 328 | 1 |
| Largest QA prompt | 1,027 | 74 |
| Longest QA answer | 695 | 184 |
| Longest book passage, with its first 128 glyphs as prefix | 128 | 513 |

All 772 target positions and four terminal boundaries are retained. No target
is truncated to pass admission. The old 3,176-question adapter subset has answers
of at most sixteen glyphs; this short-answer restriction is absent from the
new preflight. Earlier compact models already used the substantial corpus and
failed, so broader exposure alone is not claimed as a new solution.

Five CPU tests use actual native decoder layers and check pure-cache behavior,
causal masks, native zero-residual equivalence, checkpoint/plain gradients,
source freezing and serial/parallel answer inference. The full-body run then
verifies:

- All 36 answer layers receive finite nonzero residual-up gradients at every
  update. Residual-down gradients are nonzero after the zero-up initialization
  step. Query, image-readout and stopping gradients are also nonzero.
- Each of the 36 layers runs once forward and once during checkpoint
  recomputation at every update. The old frozen lower-answer cache is not used.
- Source prompt hidden states and every layer's prompt key/value fingerprints
  remain bit-exact when recomputed after training. Frozen parameters retain
  their version counters and receive no gradients.
- On the full 513-glyph continuation, changing the last physical target leaves
  every preceding predictor state bit-exact. Changing the first target leaves
  the initial query state bit-exact. Later states change, excluding a
  disconnected computation as the reason for equality.
- The zero-residual split/native maximum absolute difference is 0.000223160.
  After updates, serial/parallel short-case inference differs by at most
  0.000070572. Both satisfy the registered 0.002 tolerance; different execution
  shapes are not claimed bit-exact.

The audit opens the actual checkpoint and verifies four AdamW steps for all
85 parameter tensors, their FP32 optimizer states and shapes, complete original
training targets, prompt pixels and resource ceilings. It independently checks
initial/final change counts for the twelve writer/stop tensors, query and 36
zero-initialized up maps. Initial down-map tensors were not archived; their
updates are evidenced by the actual gradients, Adam moments and registered
execution's change checks. Complete source-body tensors are not duplicated for
an independent second before/after comparison. Prompt/frozen-body assertions
come from the bound full-GPU execution, with numerical and optimizer checks
independently audited as described.

## Parameters, cost and runtime exercise

Trainable parameters total 20,212,225: 2,359,296 new residual parameters, a
4,096-parameter query and 17,848,833 readout/stopping parameters. Deployed weights
also retain the 7,522,463,984-parameter inherited visual/contextual body and frozen
image interfaces. Training this interface does not establish a lightweight
foundation model, language acquisition from images alone, Qwen parity or a
matched-quality efficiency advantage.

Execution takes 150.811 seconds, including 20.020 seconds of pressure waiting.
Loading is 113.573 seconds; loading and waiting intervals may overlap. Minimum
available host RAM is 55.099 GiB. The measured complete cases stay below the
22-GiB allocated-VRAM ceiling. This result does not bound arbitrary longer
pages, larger batches or future optimizer configurations. The worker exits
with code zero and GPU0 returns to 16 MiB; GPU1's service is untouched.

The [runtime exercise image](evidence/visual_continuation_preflight_20261112/unqualified_runtime.png)
contains one emitted glyph before learned stopping on the short training case.
Actual emitted-pixel and feedback hashes are checked. This is an exercise of
the inference path after four updates, not held-out language evidence. There
is no new complete-answer, paired-question or book-coherence score.

## Next action

Discard the preflight updates before a full fit, retaining their checkpoint as
evidence. Prepare and register the complete training comparison from the existing
13,785-sequence source corpus, with explicit diagnostic exclusions, complete QA
answers and book continuations, data exposure, fixed snapshots and resource
limits. Preserve normal/erased inputs, both prompt layouts and every current
successful case in the evaluation. Finish whole-output review and scoring,
then confirm any development improvement on untouched source groups.

The existing usable-language goal remains active and unachieved. The current
35/62/63-page article predates this preflight and the correct-history study;
a subsequent bounded results revision must incorporate them without presenting
preflight outputs as language learning.
