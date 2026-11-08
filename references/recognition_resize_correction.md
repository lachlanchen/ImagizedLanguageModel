# Preserve the reader's native resize convention

The first complete admission run used committed implementation `627beca`.
It read 62/64 clean pages exactly (Chinese 32/32, English 30/32). Mean blank
minus clean target NLL was 9.786364 nats. Frozen-reader input gradients were
finite and nonzero for both inspected languages; peak allocation was 4.298 GiB.
No student optimization ran.

Admission failed the original preprocessing bound: maximum normalized errors
were .298058 and .281816, despite mean errors below .00324. Preserve that
[complete failed report](evidence/recognition_guided_writing/reader_v1.json).
The installed Transformers processor falls back to its Pillow backend because
torchvision is absent. Calling this a native fast-processor comparison was
incorrect. The intended two-dimensional bicubic tensor interpolation does
not reproduce clipping of Pillow's horizontal byte intermediate.

The corrected implementation explicitly selects the same native Pillow
backend and uses horizontal bicubic interpolation followed by clipping, then
vertical interpolation and clipping. It retains continuous floats for
gradients rather than differentiating through byte rounding. This is a
measured continuous surrogate, not an exact derivative of a quantizer.
No teacher weights, labels, page geometry, case order, recognition thresholds,
noise schedule or student budget change.

Repeat all 64 admissions in a fresh directory with the same original gates:
60/64 clean exact, 28/32 per language, NLL gap at least 1 nat, finite nonzero
image gradients, preprocessing mean <= .005 and max <= .08, allocation below
22 GiB. Proceed to the already declared student preflight only if every gate
passes. Retain both attempts; improved preprocessing alone is no student
language or writing result.
