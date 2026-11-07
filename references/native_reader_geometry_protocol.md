# Frozen-reader native-resolution check

This diagnostic is specified after all original full-page results, including
the unadapted HQ English check, have completed. The original unadapted source
answers 0/32 questions correctly at 1024 × 1024. No optimizer is running.

The pinned official Show-o2 MMU demo uses `image_transform`: bicubic resize
of the shorter edge to 432, then a 432-square centre crop. Its configuration
has 729 image positions. Our square 1024-page interface instead interpolates
the visual positions to 4096. This is a concrete distribution difference;
it is not yet a demonstrated cause of failure.

Apply that official geometric operation to every one of the same eighty
English input/control pages. Because all originals are square, resizing does
not crop any content. Keep the existing selection, references, instruction,
seeds, BF16 body, FP32 codec, cached greedy readout and 96-decision cap. Add
no adapters, optimizer steps, replacement cases, answer crops or teacher
conditioning. Preserve the source and transformed image hashes separately.
The changed pixels and visual positions are an explicit intervention, not a
replication of the original pixel-preserving condition.

Before interpreting answers, measure independent Tesseract recognition on all
32 resized normal pages. Reconstruct all 32 through the same sampled FP32 Wan
codec and measure recognition again. Reuse original complete-page transcript
references only for offline scoring. Never put OCR text into model inputs or
select cases from recognition outcomes. Preserve every output, including
errors and responses that hit the cap. Assess all eighty native readouts with
the existing English alternative-answer and paired-erasure scorer.

A recovery would implicate this interface change within this source/cohort.
A failure would leave source specialization, alignment, instruction placement,
small-text readability and other implementation/model limitations unresolved.
It would not establish that image-language learning is impossible or that
another checkpoint works. No result qualifies image-answer generation.

## Post-source amendment: existing adapters at native geometry

The full frozen-source 432 assessment has now completed: 11/32 exact stopped
normal answers, compared with 0/32 on the original 1024 pages. This observed
recovery motivates two further evaluation-only arms, specified before their
calls: apply the already completed 4,096-update and subsequent 2,560-update
adapter checkpoints to the same eighty resized pages. Preserve the same seeds,
references, decoding and input-only indexes. Do not retrain or choose an adapter
from these answers. These measurements test whether the existing adapters,
learned at the original geometry, preserve or damage the native-scale recovery.
They cannot measure the effect of training new adapters at 432.

Official source, pinned at `45a5a2de01d1ebd10cd5864d29310a76476cdf23`:
[transform](https://github.com/showlab/Show-o/blob/45a5a2de01d1ebd10cd5864d29310a76476cdf23/show-o2/datasets/utils.py),
[configuration](https://github.com/showlab/Show-o/blob/45a5a2de01d1ebd10cd5864d29310a76476cdf23/show-o2/configs/showo2_1.5b_demo_432x432.yaml),
[MMU interface](https://github.com/showlab/Show-o/blob/45a5a2de01d1ebd10cd5864d29310a76476cdf23/show-o2/inference_mmu.py).
