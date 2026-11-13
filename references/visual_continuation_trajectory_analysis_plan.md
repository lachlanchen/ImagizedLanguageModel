# Saved-trajectory analysis after the failed full trial

This is a retrospective analysis of the completed continuation trial, registered
before computing these new statistics. All model outputs and whole-response
readings already exist and are frozen. The original language scores and pilot
remain unchanged. No training, inference, teacher, source decision head, OCR,
font rendering or new semantic grading is permitted in this analysis.

Use every one of the 732 saved output streams at the five fixed checkpoints,
including all erasures and matched long-book controls. Authenticate each original
tensor and raster against its generation report. Inspect density, repetition and
state geometry separately from semantic quality. Empty outputs remain empty.

Measure first-eight and last-eight emitted tiles, plus every tile in each stream:
ink fraction, adjacent physical Hamming distance, exact adjacent repetitions and
foreground F1 against the constant-image baseline below. Report the fraction of
tiles above the empirical 99th-percentile ink count of complete training targets,
separately for QA and books. Include a separate count of length-at-least-32
streams; early/tail comparisons on those streams use disjoint windows.

For the saved contextual states, report relative adjacent L2 change and cosine
distance, using the same early/tail windows. For each QA pair and each normal/
erased contrast, report first-state relative L2 change, exact state equality,
first-image Hamming distance when both emit, and first-stop probability change.
These are representation measurements. Small distances or identical first images
alone do not prove semantic insensitivity or a stable dynamical attractor.

Compute a constant-image optimum for the exact balanced physical loss using all
13,766 eligible complete targets, once per source. Repeated epochs and layouts
have the same targets, so their common multiplicity cancels. For target sequence
i, let n_i be its tile count, c_ij the number of black pixels at coordinate j,
B_i its total black count and W_i=1024 n_i-B_i. With the loss's denominator clamps,

    a_j = mean_i c_ij / max(B_i,1)
    b_j = mean_i (n_i-c_ij) / max(W_i,1)
    q_j = a_j / (a_j+b_j).

This minimizes the balanced BCE within the family of context-independent images.
It is not an optimum over all conditional models. Show its probability image and
the runtime-matched q_j>=0.5 binary image. Compare saved output shapes with that
image without using it to replace or repair any output. Include zero-ink targets
and validate complete target coverage. Test the derivation against an independent
autograd calculation, including unequal sequence ink masses.

High tail ink, repetition and similarity to this baseline would support an
observational collapse description. Their absence rejects that description for
the current outputs. Either outcome leaves the cause unresolved: this analysis
does not isolate the loss, reader, timebase, image interface or feedback. An
intervention needs a separately registered unchanged control and complete-language
evaluation. The earlier failed mixture, function-transfer and clean-history
experiments remain constraints; this study does not rerun them.

Use two CPU threads, CUDA hidden, a 600-second wall ceiling, an 8-GiB peak-RSS
ceiling and a 24-GiB available-host-memory reserve. No background GUI or persistent
worker is needed. Commit the source and registration before execution; preserve
failed attempts and immutable completed reports. Bind the source, current fit,
generation reports, frozen review, corpus report and manifest by SHA-256.
