# Cold independent replay of the fixed image-only candidate

This input is fixed before the final model produces any assessment answer:
use `fresh/0000/normal`, irrespective of whether its eventual output is correct,
empty, malformed or capped. Do not select a visually favourable example.
Keep the registered 64-glyph cap and final update-18,000 checkpoint. Cold replay
tests execution independence and reproducibility, not language accuracy.

After all 72 assessment streams finish and their actual GPU process exits,
launch the standalone CLI in an empty working directory with an empty model
cache, offline library settings and the existing authenticated shared body.
Only the CLI child sees GPU0 and acquires its normal project mutex. Reuse source
weights rather than duplicating them. The verifier reads no answer references
and supplies no expected answer length, font, glyph bank or source lexical head.
The image is prepared through the actual standalone path, including its saved
observed raster. Complete tensors, decision states, stop events, feedback records
and PNG bytes must match the already captured assessment stream exactly.

Capture file opens and connection calls with `strace`. Reject training-data,
glyph-inventory, reference-string, external tokenizer-data or font-file access
and all IP connection attempts. The declared prompt image is the sole input
exception. Fixed numeric task controls and authenticated model weights are
allowed. A shared GGUF archive can contain unused vocabulary metadata; the
trace alone does not prove that those bytes are absent. The model source and
its zero-vocabulary architecture checks establish which parameters are loaded.
Do not conflate absence of runtime lexical operations with absence of lexical
metadata in the original archive.

Keep raw traces local; record their hashes and a source-bound verification
receipt. A completed cold replay of one input does not prove arbitrary-input
quality, novel-script reading, training from scratch or a smaller foundation.
The paired complete-answer review and score remain the language evidence.
