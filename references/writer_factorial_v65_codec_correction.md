# Auxiliary codec reference: native polarity correction

5 November 2026. Registered before generating corrected reference outputs.
The running four-arm diagnosis keeps all source bytes and schedules fixed.
Its main writers correctly use black=-1, white=+1. Their fitting, sampling
and measured images do not call the auxiliary V34 codec.

The auxiliary function in `scripts/diagnose_grounded_writer_v65.py` incorrectly
maps source images to ink-positive [0,1] data before V34 encoding. V34's actual
training and evaluation use white-positive [0,1] data: its loss defines ink as
`1 - target`, and its evaluator defines ink as the complement of predicted
white. Inverting both input and displayed output is not the native interface
on which the existing codec was qualified. The observed 0.813636 foreground
F1 therefore cannot be treated as native V34 reconstruction performance.

Preserve the original `codec_report.json`, `codec/` images and four-arm report
unchanged. After the owned fitting process terminates, run the separate CPU
entry point `scripts/check_v34_writer_reference_v65.py`. It must verify the
same codec checkpoint SHA and the complete fixed 64-case image cache; no new
data, optimization, checkpoint selection or threshold change is permitted.

Use native codec input `(signed_pixels + 1)/2`, black=0, white=1. Split the
same physical left/right tiles; run the unchanged FP32 EMA encoder/decoder;
convert output logits >=0 to white=+1 and negative logits to black=-1.
Retain all 64 raw images in `codec_native/`, and publish scores, all per-case
arrays and source/cache/checkpoint/implementation hashes in
`codec_native_report.json`. The original invalid-polarity output remains a
recorded implementation failure. It supplies no evidence against the earlier
codec qualification. Both old and corrected comparisons remain reconstruction
measurements on training-only images, with no language-quality claim.
