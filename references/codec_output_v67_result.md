# V67: a separate continuous output space preserves writing

The [registered audit](codec_output_v67_protocol.md) applies the original
frozen V34 EMA through `CodecBlockWriter`, without new training. It uses the
same 128 reused-development blocks as V64/V66. Physical 32-by-64 images become
two continuous 768-vectors, which the native decoder turns back into pixels.
The decoder receives neither target images nor character IDs.

| Fixed component audit, 128 blocks | Aligned codes | Shuffled codes |
| --- | ---: | ---: |
| Foreground F1 | 99.998% | 21.413% |
| Exact blocks | 126 | 0 |
| Blank blocks | 0 | 0 |

Both unchanged component gates pass: aligned F1 exceeds 90%, and the
78.584-point conditioning gain exceeds 20 points. All 128 original references,
the native encodings and all 256 output images are independently verified in
the [receipt](evidence/codec_output_v67_result.json), using the
[native-codec verifier](../scripts/verify_codec_output_v67.py).

This is reconstruction with target images supplied to the encoder. No learned
prompt-to-answer prediction is evaluated. It supports a separate writing space
instead of requiring the semantic reader's features to be invertible. Accurate
codes predicted by a causal language actor remain unproved.

The CPU audit takes 1.684 s and uses the existing 7,423,361-parameter frozen
codec. These figures exclude its earlier training and inherited language-model
pretraining. The raw run is `artifacts/codec_output_v67_20261105/`.
Report SHA-256: `058cab147a71ce40ae9fc8a327a7265ef145fa1247a9915ab9e60d29b3901a01`.
Checkpoint SHA-256: `a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33`.
