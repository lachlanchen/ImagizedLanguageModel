# Full training-source glyph-label qualification

The failed isolated reader confidently misidentifies clean reference glyphs.
Before proposing a compact training-only replacement, establish whether the
existing corpus supplies authentic, unambiguous labels for its actual cells.
This is a CPU data/identifiability check, not verifier training, student training
or a language result. The completed saved-preprocessing audit does not repair
the rejected reader.

## Fixed scope and information boundary

Use every one of the original 13,785 training sequences and every prompt and
target cell in their packed training shards. Preserve the 19 records excluded
from the later continuation fit; this label-support audit does not redefine
that fit's source set. Bind the corpus report/verification, all training pixel
and source shards, original raw-source digests, Noto font/face, original source
selector/renderer, new checker and tests before deriving labels. Commit and push
the registration before the run. Read no model weights or teacher-unit arrays.

Reconstruct source ownership through the original book/CMRC selector. That
selector uses official-development titles/contexts and the existing natural
partition to exclude overlap, preserving the established split. Only records
actually marked training contribute character labels or images. No development
reference image, calibration output, confirmation source or development answer
enters the label inventory. Raw files used for original partition screening are
not a second source of training labels.

For every corpus record require the same source ID, group, complete text hash,
prompt boundary and glyph extent. Re-render each source character with the
original font and compare **every original cell byte**. Preserve counts of any
disagreement and retain complete records; do not fix, crop, replace or drop
positions. Construct the inventory from actual mask/codepoint pairs, preserving
all occurrences and every conflicting label. Re-rendering is a verification
control, not a replacement for the saved original masks.

## Prospective conditions and decision

Require complete authenticated coverage, zero renderer/pixel disagreements,
one raw canonical character per nonwhite mask and only whitespace source labels
at physically white cells. Report all nonwhite aliases, source codepoints and
counts. Preserve raw whitespace labels separately; their identical physical
appearance does not recover newline/tab layout. Nonspace invisible characters
are a separate failure. Apply no NFC/NFKC or visual-homograph merger to make
the uniqueness condition pass. Duplicate appearance under different raw labels
disproves a perfect context-free raw-character decoder, not visual-language
learning generally.

Retain a complete per-record label stream and an indexed unique physical-mask
inventory privately with hashes. A mismatch, missing owner, conflicting
nonwhite label or invisible nonspace label rejects exact context-free source
supervision. Preserve every failure and choose the next representation/supervision
decision from it. A pass only establishes source support; it does not authorize
verifier training, qualify recognition on emitted images or admit an actor fit.
Those require separate registrations and all original complete-language tests.

Use one CPU worker under its exclusive audit lock, two CPU threads, at most
900 seconds and 2 GiB process RSS. Retain 24 GiB available host RAM and 48 GiB
free disk; derived data must remain below 1 GiB. No GPU, model API, new service,
weight download or automatic retry. Preserve partial data and an error receipt
on an execution failure. The complete objective remains independent image-based
language generation, including historical and unencoded writing.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OPENBLAS_NUM_THREADS=2 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m unittest tests.test_glyph_label_audit -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OPENBLAS_NUM_THREADS=2 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/audit_training_glyph_labels.py register
# Commit and push the fixed registration before deriving labels.
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OPENBLAS_NUM_THREADS=2 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/audit_training_glyph_labels.py run
```
