# Adapt visual continuation while preserving prompt reading

Status: direction selected after the correct-image-history comparison;
not implemented, registered, fitted or qualified.

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

## Candidate to implement and test

Keep source visual prompt prefilling frozen. Test trainable continuation
adapters activated after that prefill, including a learned continuous first
answer query, at more than the current four final contextual layers. They
condition on the prompt memory and actual preceding glyph images. The neural
image writer and learned stop decision still produce the answer. This tests
adaptation of the state transition to visual steps, rather than pretending that
one glyph is one native source unit.

For a candidate residual adapter, write the transition schematically as
\[
 h^{\ell+1}_t = F_\ell(h^\ell_t; C_\ell(x), h^\ell_{<t})
                 + g_t B_\ell A_\ell h^\ell_t,
\]
where source prompt memory \(C_\ell(x)\) is fixed, \(g_t=0\) during prompt
prefilling and \(g_t=1\) for the answer query and visual continuation. The
low-rank maps are training parameters, not a glyph inventory. This specifies
the intervention to implement; it does not establish a successful loss,
memory footprint or language result. The query alone is not the proposed repair.

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

## Required next work

Implement one continuation-gating design and a real preflight on training cases.
Verify disabled-prefill identity, causal exclusion of the current/future target,
nonzero gradients through every selected trainable answer layer and the writer,
and unchanged frozen weights. Measure actual allocated memory and wall time on
GPU0 under the existing mutex and 22-GiB ceiling. Do not claim feasibility from
parameter counts or a mock network. Preserve any failed admission or memory test.

Register a single full comparison only after this preflight passes. Select and
account for authentic full training sequences from the existing corpus; keep
whole-work and passage-group splits and exclude all diagnostic development
records from training. Establish exact data exposure, optimizer steps, snapshots,
resource ceiling and failure action in advance. Preserve the parent, balanced
and flow controls, including all eight union proportional exact answers and
both fixed exact answers.

Judge the model with complete readable image answers, paired question dependence,
book continuation, natural stopping and erased-input controls. A pilot needs
untouched confirmation; copying and reference-conditioned writing remain separate.
If this candidate fails its preflight or full language gates, record that result
and change the hypothesis. Do not extend an unchanged failed fit or replace the
full independent visual-language objective with a small-head demonstration.
