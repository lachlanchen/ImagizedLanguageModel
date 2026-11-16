# The fixed image-encoder metric fails complete-answer qualification

The fixed frozen-encoder metric accepts **six of nine known-correct answers**,
zero of nine readable-wrong answers and zero of 46 uncertain answers on all 64
preserved outputs. Correct/wrong pairwise AUC is .9753. Coverage, finite distinct
features and cap handling pass, but the all-correct acceptance and strict reward
separation requirements fail. The [registered metric](frozen_encoder_verifier_protocol_20261116.md)
is rejected. No student update or policy fit follows it. Do not tune another
normalization, layer, checkpoint or radius on these outputs.

The [complete evidence](evidence/frozen_encoder_verifier_calibration_20261116.json)
records 9,581 training-source prototypes, 297 actual stopped-output cells and
197 forward calls through the fixed 18,486,528-parameter image encoder. It uses
FP32 forward arithmetic and direct FP64 feature distances. Measured calibration
time is 8.771 seconds, whole-action time 14.980 seconds, peak allocated GPU
memory .688 GiB and peak RSS 1.065 GiB. These are component costs, not a language
model efficiency result. Original pixels, uncertain readings, old scores and
all earlier rejected judges remain unchanged.

## Next learning comparison

A qualified reward is unnecessary for direct supervision from authentic source
images and saved training-only teacher states. The compact full-rank model's
frozen contextual body is the selected variable for a new matched comparison.
The [full-context execution preflight](trainable_visual_context_preflight_protocol_20261116.md)
retains its reader, image writer and six source objectives, and changes whether
all 1.409B contextual parameters learn. Both arms use the same FP32 parameters
and host-moment AdamW. A successful execution check admits preparation of a
separately registered complete-corpus comparison; it does not establish language
learning or authorize a full fit by itself. Earlier trainable-body failures
remain constraints. The new branch uses no rejected reward, development output
as a fitting label, runtime lexical head or reference-answer repair.
