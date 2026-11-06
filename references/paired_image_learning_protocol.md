# Paired-image learning: execution and writing checks

Specified before local base-model optimizer execution, 7 November 2026.
The goal remains a useful, independent visual language model. The completed
zero-shot editor failure is not a training result. Neither a successful
backward pass nor a small training fit qualifies the general ILM objective.

Use the pinned undistilled FLUX.2-klein-base-4B revision and fixed generic
condition described in [the learning direction](paired_image_learning_direction.md).
Retain the exact source VAE posterior mode, patchification, batch normalization,
packing, spatial IDs and native sampler. Answer features are present only in
the supervised noise-time objective; autonomous inference receives an input
PNG, the fixed condition, source weights and optional learned adapter. It
must not load the training manifest, answer cache, OCR or a font renderer.

Preparation renders 128 training and 32 validation literal-copy examples,
half Chinese and half English. Strings are different across both splits.
Random Chinese strings test exact visual composition, not linguistic fluency.
It also selects 256 training and 32 validation natural Chinese questions from
the existing CMRC training source, paired by passage with different answers.
Articles are partitioned by a deterministic title hash before selection.
These public validation examples are held out from this optimizer, not claimed
to be a pristine project-wide sealed test. Render full source passages and
questions at 32-pixel Noto Sans CJK type on 1024-square input pages; independently
fix output canvases to 512 by 256. Reject any pair that clips at that geometry,
record exclusions, and never crop a passage around its answer. This geometry
is an initial finite-canvas experiment, not a solution for arbitrary-length
books or historical/handwritten forms.

Adapt native block attention/feed-forward and input/output linear projections
with rank-16, alpha-16 low-rank residuals. Keep source weights frozen. Adapter
masters and AdamW state use FP32 with BF16 native computation and gradient
checkpointing. For sigma uniform on [0,1], regress epsilon-minus-answer from
the native mixture (1-sigma)*answer + sigma*epsilon. All examples receive
the same generic numeric condition, with no varying question caption.
This is conventional conditional flow matching, not a novel theorem.

First execute two real optimizer updates with batch one, rate 1e-4 and
AdamW weight decay 0.01; record finite/nonzero gradients, actual parameter
changes and peak allocated VRAM. This checks actual model execution within
one GPU. It does not test semantic improvement or writing quality. The
standalone CLI must also produce unrepaired autonomous answer PNGs in a
fresh inference process using no training data.

The first bounded writing experiment uses the first 32 training copy images,
512 updates, batch one, accumulation one, seed 20268321, rate 1e-4 and the
same optimizer. Fix before/after display cases to the first four training
and first four validation copy images in manifest order. Infer using the
same base source, 50 native steps, guidance one and matching seeds; comparison
to the earlier distilled zero-shot model is not a matched training effect.
Guidance one evaluates the same conditional field that receives supervision,
without extrapolating against an empty-condition branch that this recipe
does not train. This sampler choice is fixed before optimizer execution.
Retain all eight before/after outputs. A loss decrease alone cannot qualify
writing. If the initial display is promising, assess all 32 training copies
and all 32 validation copies: at least 30/32 of each must contain the exact
requested string alone as legible writing. Calibrate offline OCR on every
reference and retain uncertain cases for raw-image review. Failure rejects
scaling this writing recipe without a diagnosed correction.

Natural QA optimization, held-out paired-question answering, erased-question
controls, style/length transfer, historical forms and matched-quality efficiency
remain required after writing. Passing new-string copying is a necessary
diagnostic, not evidence that those broader requirements have been achieved.
