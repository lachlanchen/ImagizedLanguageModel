# Blinded transcript diagnostic for V61 answer images

Registered 2026-11-05 while production training is at update 970/1,000,
before any V61 development answer images or scores have been inspected.
This is auxiliary evidence and does not alter the frozen V61 decision gates.

Exact raster identity can reject a readable answer with a few changed pixels.
After the locked image audit, assess modern Chinese answer content separately
with the existing CPU Tesseract engine, languages chi_sim+eng, LSTM OEM 1,
single-line PSM 7. Pin the executable version and language-file hashes.
Upscale each generated answer strip fourfold with nearest-neighbour sampling
and add a 16-pixel white border. The recognizer sees **only the generated
answer**, never the question, passage, reference or model name. A zero-cell
answer has empty transcription without invoking OCR. No correction, language
model call, reference-driven alignment, glyph snapping or alternative OCR
selection is allowed.

Use all fixed 256 primary outputs in trained, trained-blank, base and base-blank
conditions. Calibrate on the first retained reference images from those same
rows, without changing preprocessing after seeing calibration. Accept all
retained reference strings when scoring model transcriptions. Normalize NFC
and remove whitespace only; preserve punctuation and case. Report string
exact match and minimum reference-normalized Levenshtein error, plus bounded
character accuracy max(0,1-error). These are transcript diagnostics, not the
official CMRC segmented F1. Preserve raw transcripts/confidences and images.

Bootstrap trained-minus-base and normal-minus-blank by source article, 2,000
draws, seed 20266319. Report gold-image OCR error and the limits of a noisy
recognizer. No confidence-threshold selection, training update, new pass gate
or claim about historical/unencoded characters follows. The primary image
audit remains unchanged even if this diagnostic finds readable language.
