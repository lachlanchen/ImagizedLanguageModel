# V54 context attribution: post-result diagnostic plan

Date: 2026-11-05. This plan is written after the failed locked V54 audit and
before running the diagnostic. It does not amend that audit or its decision.

Question: does the exact raster density use the complete visual history,
only the short suffix retained by the shuffle, or mainly already-revealed
strokes within the target? A positive effect does not itself establish coherent
generation. The purpose is to choose the next learning intervention without
another unmotivated writer redesign.

Use the unchanged production checkpoint and the same 512 deterministic
development windows, source grouping and audit seed 20265420. Keep model,
training and primary evaluator sources unchanged. No optimization, sealed
partition, image selection, temperature search or new threshold is allowed.

Record per-pixel and per-row teacher-forced NLL under:

1. ordered natural hidden state and predicted field;
2. suffix-preserving earlier-history shuffle;
3. complete hidden/field swap with a different source record;
4. zero state and field (out-of-distribution intervention);
5. supplied true image field and zero hidden state (answer-revealing control);
6. supplied field from a different source target and zero hidden state.

For the complete swap, sort window indices by source record and cyclically
shift the sorted list by half its length. Require every donor source to differ
from the target source. Use the same donor for the supplied-image derangement.

Report summed NLL, per-row curves, four fixed eight-row bands, and paired
source-record bootstrap intervals (2,000 draws) for shuffled/deranged minus
ordered. Compare reconstructed ordered/shuffled NLL with the saved primary
per-example values; fail on a numerical mismatch. Preserve all raw rows and
hashes. These are model-specific likelihood interventions, not calibrated
mutual-information estimates or independent confirmatory tests. Pixel
autoregression legitimately observes earlier target pixels; their explanatory
power must not be mistaken for information supplied by language history.
