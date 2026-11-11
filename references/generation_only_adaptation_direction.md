# Adapt visual continuation while preserving prompt reading

Status: implemented and real full-body preflight passed. The full training
comparison is not registered or fitted; usable language remains unqualified.
See [the complete preflight result](visual_continuation_preflight_result.md).

## Why this is the next model question

The [completed comparison](reference_image_history_result.md) finds much stronger
later writing with correct proportional image history, but both physical gates
still fail and initial writing remains weak. Several nearly exact initial forms
continue readably; mere state divergence is not the failure criterion.
The [earlier feedback ceiling](glyph_feedback_ceiling_result.md) independently
shows that first-character steps alter the source policy even with exact vectors.
Its cohort and helper outputs must remain separate from the current language test.

The current contextual adaptation changes only four final layers, with fixed
image feedback and cached lower-layer answer states. Its success on another
cohort and failure on the present one do not establish that this restricted
adaptation has learned reliable glyph-step language dynamics. Conversely,
changing an output density alone is not a new untested idea: the completed
[whole-image mixture](correlated_native_density_v62_result.md),
[first-function transfer](glyph_first_function_transfer_result.md), compact-span,
decoded-loss and direct-context flow results must be read before choosing a fit.
Do not repeat them as new solutions or project their scores onto this cohort.

## Implemented candidate

The implementation freezes source visual prompt prefilling and activates
trainable continuation adapters after it, including a learned continuous first
answer query, at all 36 contextual layers. They
condition on the prompt memory and actual preceding glyph images. The neural
image writer and learned stop decision still produce the answer. This tests
adaptation of the state transition to visual steps, rather than pretending that
one glyph is one native source unit.

For a candidate residual adapter, write the transition schematically as
\[
 u^\ell_t = F_\ell(h^\ell_t; C_\ell(x), h^\ell_{<t}),\qquad
 h^{\ell+1}_t = u^\ell_t + g_t\frac{\alpha}{r} B_\ell A_\ell u^\ell_t,
\]
where source prompt memory \(C_\ell(x)\) is fixed, \(g_t=0\) during prompt
prefilling and \(g_t=1\) for the answer query and visual continuation. The
low-rank maps are training parameters, not a glyph inventory. The implementation
places the residual after each native layer, with rank eight and alpha sixteen.
Its measured footprint is 18.123 GiB for the registered complete cases; a
successful language objective remains unproved. The query alone is not the proposed repair.

The generation gate must leave source prompt states and memory bit-exact when
its adapters are disabled during prefill. It must permit the first output state
to adapt without changing the prompt pixels or supplying an answer. The query is
a learned numeric vector, not a lexical ID or reference-conditioned control.
No source lexical head, character inventory, OCR, font or teacher enters deployed
answer generation. Inherited weights and offline supervision remain disclosed.

Only frozen prompt memory may be cached. If answer-layer adapters or the feedback
encoder change, recompute the affected answer computation during training;
reusing the old frozen lower-layer answer states would test another restricted
head fit, not the proposed transition adaptation. Keep causal masks, coordinates,
output pixels and terminal decisions explicit. Output sampling and density must
be fixed before the fit, not selected after reading development answers.

This is a hypothesis about adaptation capacity and sequencing. It is not proof
that the source body's language is inaccessible, that all layers are needed,
or that this change will outperform the retained model.

## Completed prerequisite and next work

The complete preflight and independent optimizer/data audit pass: disabled-prefill
identity, causal target exclusion, nonzero answer-layer/writer gradients and
frozen-weight checks are verified by the registered execution. Peak allocation
is 18.123 GiB for the measured complete cases. Preserve this evidence and do not
repeat the preflight as pending work.

Next prepare and register the single full comparison from the substantial
authenticated corpus. Discard the four preflight updates before the full fit. Select
and account for authentic full training sequences from the existing corpus; keep
whole-work and passage-group splits and exclude all diagnostic development
records from training. Establish exact data exposure, optimizer steps, snapshots,
resource ceiling and failure action in advance. Preserve the parent, balanced
and flow controls, including all eight union proportional exact answers and
both fixed exact answers.

Judge the model with complete readable image answers, paired question dependence,
book continuation, natural stopping and erased-input controls. A pilot needs
untouched confirmation; copying and reference-conditioned writing remain separate.
If the full comparison fails its language gates, record that result
and change the hypothesis. Do not extend an unchanged failed fit or replace the
full independent visual-language objective with a small-head demonstration.
