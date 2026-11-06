# CPU transcription calibration for saved native image answers

This auxiliary diagnostic is specified after inspecting the 1,000-, 5,000-
and 10,000-update training images, before running this recognizer on the new
calibration or actor images. It changes no training parameter, saved answer,
physical-image metric or original evaluation gate. The snapshots are reused
training diagnostics, not held-out evidence.

Use the installed Tesseract executable with `chi_sim+eng`, OEM 1, PSM 7 and
TSV output. Record the executable, language-weight hashes and version before
recognition. Use two CPU workers, one OpenMP thread per worker and a 45-second
request limit. Require `CUDA_VISIBLE_DEVICES=''`; do not start a GPU model or
contact the shared LocalLLM service. Preserve every raw TSV, error and parsed
response. Do not retry failed requests or tune settings from the results.

Select calibration images from the authenticated complete-line training cache
and the 12,976-record admitted image-code cache. Select 128 distinct Chinese
training articles, one admitted question per article, and sixteen admitted
English records from each of the eight Dolly categories. Sort groups and
records, then sample without replacement using NumPy seed 20268010, Chinese
first. Reuse the actual complete target pixels, removing only the two declared
white termination blocks. The expected label is the exact source string that
produced that target. Do not read development or sealed-test examples.

Add sixteen blank strips and sixteen independent random-ink strips, lengths
one through sixteen 64-pixel blocks, ink probability 0.15, seed 20268011.
Enlarge all inputs fourfold by nearest neighbour and add a sixteen-pixel white
margin. Keep every source pixel. Give the recognizer only anonymous image
filenames; source questions, expected strings, model identity and conditions
are excluded from its arguments.

Normalize expected strings and transcripts with NFC and whitespace removal
only. Retain punctuation and case. Calibration requires at least 95% exact
reference transcripts separately for Chinese and English, all sixteen blanks
empty, at least 90% of random-ink controls empty and successful, parseable
recognition for every control. Record source-specific validity; an unqualified
source cannot receive actor transcript scores.

Freeze the complete saved-diagnostic inventory at startup. If Chinese
calibration passes, transcribe every Chinese answer from those completed
snapshots, including empty writing, without repairing pixels. Validate raw
tensor and preview hashes before use. Compare transcripts with all retained
source reference strings using exact normalized agreement, normalized edit
distance and bounded character accuracy. This is auxiliary transcript
agreement under a calibrated reader, not ground-truth semantic understanding.
If calibration fails, make zero actor-recognition calls. If any actor request
fails, retain it and withhold the complete snapshot aggregate; do not drop the
case or substitute an empty transcript. Empty actor tensors, in contrast, are
known empty writing and need no recognition call.

The experiment may run while GPU training continues because it uses only
bounded CPU image decoding and OCR. The final independent image actor never
uses this recognizer. A later neural-reader diagnostic requires separate
calibration after the GPU training and original audit finish.
