# Probability diagnosis of the joint writer's empty images

The full-corpus joint writer fails its frozen physical gate: development books
give 91/108 clear nonblank selected forms and training fixed-layout QA 90/101.
All seventeen uncredited selected development-book forms are actual empty
images; four corresponding source images are also empty. Preserve the verdict,
all 33 board receipts and all 2,080 frozen readings. This diagnostic cannot
qualify that actor or authorize its blocked autonomous comparison.

Before another fit or a decoding change, score the exact final density on all
12,597 target positions in the 124 already prepared conditional cases. Reuse
the immutable FP32 predictor states, physical targets and saved emitted pixels.
There is no reader replay, optimization, new image generation, teacher or OCR.
The only new neural computation is teacher-forced probability evaluation of the
source image, empty image and retained emitted image under the same fitted head.
The source image is diagnostic evidence, never a runtime candidate supplied to
repair an answer or a frozen transcription.

For zero-history log probabilities `l[j,v]`, compute

    log p(empty | h) = sum_j l[j,0]
    U = max_j {sum_(k<j) l[k,0] + max_(v>0) l[j,v]}.

Every nonempty image first differs from the empty image at some position `j`.
Its later probability factors are at most one, so its complete log probability
is at most `U`. Thus `U < log p(empty)` certifies the empty image as a global
mode in exact arithmetic. Failure of this loose bound proves nothing about
the best nonempty image. Separately, a greater source-image probability is a
constructive counterexample to the empty greedy choice being the whole-image
mode. It is not evidence that beam search will find that image or that its
canonical continuation is the only semantically valid one. Use a fixed 0.001-nat
margin with FP32 neural scores and FP64 accumulation; this is numerical evidence,
not a formal floating-point certificate.

Retain every case and position, including true source whitespace. Compare the
saved empty outputs with zero-prefix local argmax choices to expose numerical
inconsistency rather than assuming decoding equivalence. Preserve all source,
empty and emitted log probabilities and the nonempty prefix bound. Report
complete-case and selected-position results separately.

At least seven selected development-book false blanks and one selected
training-fixed-QA false blank must prefer the source by the fixed margin before
a search-only intervention is considered: those are the minimum gains needed
to cross the two failed group thresholds. Meeting this condition permits a
separately registered decoding comparison; it does not select a search width,
change the old gate or establish readable language. If it fails, do not assume
that a larger beam repairs the density; revisit conditional prediction and
its training objective before a new bounded model comparison. Any subsequent
language trial retains complete answers, paired questions, erasures, long books,
retention, stopping, untouched confirmation and whole-model accounting.

Commit this source, tests and protocol, then create and commit/push the actual
registration with `scripts/analyze_joint_writer_blank_density.py register` under
`CUDA_VISIBLE_DEVICES=''`. Only then run `run` with `CUDA_VISIBLE_DEVICES=0`.
The exclusive GPU0 mutex, 24-GiB available-host reserve, 2-GiB allocated-VRAM
ceiling, two CPU threads, 900-second total budget and 120-second pressure limit
apply. There is one attempt and no automatic retry. The original 45 sources,
reader, head weights, corpus, physical grades and autonomous scores stay intact.
