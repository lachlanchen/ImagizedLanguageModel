# Decision after the completed full-context comparison

The [completed comparison](trainable_visual_context_result_20261117.md) rejects
the six-objective visual-span recipe in both arms. Treatment uses a fully learned
1.409B contextual body and still has no readable complete QA or coherent complete
long-book continuation. Both full fits, audits, all 792 generations, both separate
cold checks, all visual readings and original scoring are terminal. Preserve
their sources, checkpoints, raw images, uncertain readings and failed executions.
Do not repeat their generation or repair their readings.

This rejects the tested recipe, not every pixel-language architecture. Better
stopping without readable writing is an observed difference; semantic learning
and conditional glyph choice remain unestablished. The earlier first-span and
correct-history diagnoses constrain explanations but do not identify the cause
in this newly trained context. Repeating the completed controls on their old
actors will not resolve this comparison.

## Next bounded investigation: competing training signals

The implemented objective sums six losses: next-span flow, next-span length,
feedback vectors, feedback counts, span boundaries and teacher-state cosine.
Equal scalar weights do not imply equal gradients. Source-feedback regression
and teacher-state matching constrain internal coordinates but do not themselves
specify the next visible answer. The full comparison shows that allowing the
body to move under this combined signal did not produce usable writing.

The next diagnostic is a **training-only gradient accounting with no optimizer updates** on the
two exact final actors and authenticated complete training records. Its changed
measurement is the decomposition of their combined training signal. It must be
separately registered before execution, with fixed records, parameter groups,
checkpoint/projection identities, numerical precision and resource limits. No
development answers, DRCD records, new preference judge or policy fit is involved.
No optimizer update or retained fitted model is permitted.

For each reader/context/writer parameter group, measure the conditional-flow
gradient g_F and the sum g_A of the other gradients. Record their norms, ratio,
inner product and cosine, including zero-gradient cases. Preserve component
losses and all nonfinite failures. This tests a specific possible source of
interference rather than assuming that reduced loss implies semantic progress.
Use complete records, not isolated glyph copies or cropped target answers.

For ordinary gradient descent with a positive scalar clip c, the first-order
change in flow loss would be

\[
 \Delta L_F\simeq-\eta c\,g_F^\top(g_F+g_A).
\]

This expression is **not the AdamW update law**. Negative gradient cosine alone
does not establish harmful actual updates. If examining AdamW, use the retained
moments, the registered global clipping rule, group learning rates, bias
corrections, epsilon and weight decay to compute the complete hypothetical
direction without writing it to parameters. Report that it is a diagnostic
counterfactual at a new training batch, not a replay of an original update.
Do not infer actual loss changes from a first-order approximation alone.

Resource admission must retain the existing one-owner GPU mutex, 48-GiB host and
20-GiB free GPU entry conditions, 24-GiB host/22-GiB allocated GPU runtime limits
and foreign-service protection. Stream gradient summaries to host memory instead
of retaining six full gradient copies on the GPU. A separately bound support
check must establish that the chosen full records and measurements fit those
ceilings; do not silently shorten records after a resource failure.

## Actions for either result

If auxiliary signals consistently oppose the conditional image objective under
the relevant update direction, prepare one prospective objective intervention
and unchanged control. A training-only update/reset preflight must then show
actual conditional learning on complete authentic records before any new full
fit is admitted. Its acceptance rules must include complete writing rather than
only gradient signs or lower auxiliary losses.

If interference is absent, inconsistent, or insufficient to explain the failure,
reject that explanation. Do not select new loss weights from development
outputs. Use the measured limitation to choose the next single change; no new
algorithm is presently qualified by this decision. Likewise, neither finding
alone is a language result or a reason to rerun the full corpus unchanged.

The full engineering goal remains active. A working algorithm still needs
question-dependent complete image answers, coherent naturally stopped books,
untouched-source confirmation, execution independence and whole-model resource
accounting. Historical forms and etymology remain part of the full objective.
