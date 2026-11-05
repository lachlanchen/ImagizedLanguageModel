# V68: execution of the fixed autonomous development audit

This implementation follows the already registered V68 protocol and the
source-exact 192-case preparation. It changes no training source, objective,
case selection or threshold. The training process must complete all 10,000
updates and release GPU 0 before an evaluator can load the final adaptation.
Do not run a second owned model job while training is live.

The [pre-generation layout correction](codec_language_v68_layout_control_correction.md)
fixes a measured control confound before any answers are generated. Fold the
normal prompt once and place its blank control in the identical source ranges;
erasing ink must not reflow the page. Preserve the original CPU preflight and
use corrected `audit_preflight/attempt02`. All cases, references, metrics and
quality gates below remain unchanged.

Pin prepared-data report
`4f90406b51670e4e82327d1396e1c954e62bfb5d7b31c437b5c59e88a9180fe5`.
Run all 192 questions under normal and blank controls: 384 actual image
generations. The actor receives an owning, folded prompt-image tensor alone;
it receives no reference, recorded width, case ID or source text. The maximum
capacity is the fixed 128 physical blocks for every case. Retain the learned
stop events and all unmodified generated blocks, including ordinary whitespace.

Read references only through the separate scoring interface. Foreground F1
aligns physical strips at the left edge, counts all predicted/reference ink,
and scores intersection in the overlapping width. Missing/extra foreground
therefore contributes error. Exact reference agreement additionally requires
identical full physical dimensions and a learned stop. High F1 with a missing
white tail cannot pass exact agreement. Compare against every fully rendered
recorded reference, preserving each candidate score.

Use the six Boolean checks implied by the frozen protocol: Chinese exact >=.5,
foreground F1 >=.8, both-correct pairs >=.35, normal-minus-question-blank F1
>=.1, strictly positive article-bootstrap lower bound, and normal learned-stop
rate >=.9. Bootstrap 64 distinct two-question articles, 2,000 draws, seed
20266831. English scores are recorded-reference agreement only. Neither this
development test nor the actor's inherited foundation establishes general
language or historical-writing capability.

Before final weights exist, a CPU-only evaluator preflight verifies every
prepared file, pixel-domain invariants, prompt storage ownership, exact/stop/
width distinctions and the article-cluster calculation. It makes zero model
calls and reports no model quality. Keep this receipt separate from the final
generation directory.

Freeze evaluator/runtime/scoring sources before generation. Bind the final
training report, adaptation checkpoint, prepared data and source hashes in an
immutable audit identity. Persist each completed raw generation atomically,
with prompt-pixel digest and case/route identity. Resume a partial audit only
under the same identity, without re-generating committed cases. Retain binary
PNG output when the generated strip is nonempty; an empty answer has a real
zero-width tensor and no synthetic placeholder image. Never repair outputs.

Independently recompute all scores from the raw tensors/PNGs and original
references after generation. A source-matched loader may reuse the local
authenticated foundation archive at startup; it makes no source-model call.
This is not the final standalone bundle. Offline deployment isolation remains
a separate requirement after useful language qualifies.
