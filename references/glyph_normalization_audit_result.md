# Mean removal does not explain the observed first-form gap

The [fixed CPU audit](glyph_normalization_audit_protocol.md) reuses 128 already
scored first-answer states across four conditions and two cohorts. It performs
no model or writer forward pass, creates no image and updates no parameter.
The original source matrix acts only as an offline diagnostic.

Subtracting each state's unweighted mean changes **zero of 128** source-unit
winners. First-character reference matches remain 30/32 and 31/32 in the first
cohort, and 29/32 in each condition of the second cohort. A weighted-mean
reflection preserves both the final RMS shell and the subsequent LayerNorm
output; it also changes zero observed winners. Maximum numerical shell error
is 1.82e-12 and LayerNorm error is 1.07e-14 in the double-precision check.
Contrived reflected states are not proven reachable by real image prompts.

This audit supplies no evidence that mean removal caused these answer failures.
It does not prove normalization harmless for every possible state. The next
writer experiment therefore retains the current architecture rather than
claiming a normalization repair. The result is retrospective, uses CPU FP32
projection of BF16-rounded source weights, and does not measure complete
language accuracy. Execution takes 120.12s with no CUDA initialization.

The [complete report](evidence/glyph_normalization_audit_v1/report.json)
retains per-state projections, exact input hashes and numerical checks.
