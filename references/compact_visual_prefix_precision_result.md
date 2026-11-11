# Numerical consistency of one compact visual prefix

A registered frozen-weight diagnostic stopped when a separately encoded
prefix and a full causal pass differed by more than its fixed 0.02 relative
RMS tolerance. The stopped attempt retained 25 complete V1 case reports and
four reference arms of the following case. It made no optimizer update;
the wider model had not loaded. Preserve these partial outputs separately
from a completed diagnostic comparison.

A second, bounded probe isolates the failing development-book case: 312
prefix glyphs, 305 native updates and 632 glyphs in the full record. The probe
uses the final V1 checkpoint and the same image encoder and codec. It makes
twelve contextual-body calls and 24 velocity calls, with no training.

| Comparison | Relative RMS difference | Interpretation |
| --- | ---: | --- |
| Codec prefix versus full-record batch | 0.0000003852 | Small batch-dependent numerical difference in the continuous codec output. |
| Image encoder, same latents in prefix versus full batch | 0.0000004703 | Small numerical difference before the contextual body. |
| Identical prefix vectors, BF16 full versus truncated body | 0.050291 | The tested full-sequence and prefix-only BF16 evaluations differ. |
| Prefix-only mask/cache flag change, BF16 | 0 | Bit-equal for the tested prefix. |
| Same-shape future-input erasure, BF16 | 0 | Bit-equal prefix state; no future leakage observed in this intervention. |
| Native vectors, BF16 full versus truncated body | 0.013023 | The discrepancy is also present, though smaller, with native inputs. |
| Identical prefix vectors, FP32 full versus truncated body | 0.000002592 | Higher-precision arithmetic nearly removes this discrepancy. |
| Same-shape future-input erasure, FP32 | 0 | Bit-equal prefix state in this intervention. |

The FP32 run casts the same BF16-rounded contextual weights to FP32. It does
not recover unrounded source weights or train a new model. Float32 matrix
precision is `highest`, with TF32 disabled. These measurements identify a
sequence-shape/precision-dependent numerical discrepancy in this one case.
They do not establish a general bound or the cause of the language failure.

The reference next span contains punctuation and whitespace. Its three
BF16-conditioned generated images all fail exact glyph reconstruction. The
[raw images](evidence/compact_visual_prefix_precision_20261111/) retain this
failure; their foreground F1 differences are not evidence of language ability.
The [receipt](evidence/compact_visual_prefix_precision_20261111.json) binds the
case, source, checkpoints, measurements and original PNG hashes.

The probe takes 32.51 seconds, including 10.01 seconds waiting for memory
pressure, with 5.530 GiB peak allocated GPU memory. The preceding stopped
attempt takes 125.07 seconds. Both costs remain in diagnostic accounting.

The replacement comparison uses identical prefix-prefill and cached update
schedules for reference-image and generated-image history. Reference contexts
receive the correct image spans one at a time with supplied counts; the
native bypass receives the matching native updates. The original consistency
tolerance stays fixed. Full-record reader measurements remain separate.
Cases, seeds, checkpoints, sampling steps and other interventions are unchanged.
See the [diagnostic plan and amendment](compact_visual_span_failure_localization_plan.md).
