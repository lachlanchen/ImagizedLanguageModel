# Complete copying observation after padding reduction

Specified before the remaining model calls, 7 November 2026. The completed
512-update strip arm produces 2/4 training and 1/4 validation OCR exact
matches in the original fixed screen. Raw inspection finds improvements,
omissions and malformed letters. No reliability or language criterion is
claimed passed. The screen warrants completing the originally declared
32-training/32-validation copying observation.

Use the same strip checkpoint, source, condition, input preprocessing, 50
steps, guidance one and 512-by-256 output. The four original training and
four original validation outputs are immutable observations. Reuse their
actual PNGs, tensors, hashes and seeds rather than regenerate them. Generate
only the other 56 input images. The full cohort is the first 32 training
copies followed by all 32 validation copies in the original prepared manifest.
For an original fixed case retain its existing seed; for every other case at
zero-based full-cohort position i use seed 20268421+i. Fix both complete and
remaining input-only indices before generation. No expected string, target
image or question-specific text condition enters the fresh generator process.

Aggregate the eight existing and 56 new observations with their actual paths
and source hashes. Assert exactly one output for every declared ID, matching
the same checkpoint, condition, canvas and sampler. Report 64 logical outputs
from eight retained calls plus 56 additional calls, not 64 new generations.
Use the original separately calibrated OCR and all expected reference strings;
retain every output and inspect complete galleries. Original calibration
recognizes 27/32 training and 28/32 validation references, so OCR exact match
is an exploratory observation and uncertain readings require the pixels.

The unchanged requirement is 30/32 legible exact requested strings alone in
each split. A complete copying observation does not qualify language learning.
Question answering, question interventions, longer writing, historical forms
and meaningful efficiency at language quality remain outstanding.
