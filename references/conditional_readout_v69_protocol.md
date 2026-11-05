# V69: conditional writing recoverability from the frozen actor

Registered before selecting examples, capturing features or fitting heads.
The completed V68 actor fails before generated history and on training cases.
This diagnostic asks whether a different linear readout can recover accurate
first-block writing from the same frozen prompt states. It does not qualify
complete answers or general language, and cannot prove absence of information
if the tested readouts fail.

## Fixed sources and sample boundary

Use the unchanged final V68 adaptation, SHA-256
`39d978ae7eefa4fff5a73084e7dd5b262ceb09ecd09b253e2d3627fe03a4b66c`,
the qualified V34 writing codec and the authenticated inherited actor. Preserve
the original failed autonomous report and verification receipt. No existing
implementation or completed artifact is changed. Use only physical GPU 0 for
feature capture, after checking memory, existing jobs and sessions. Fit and
verify on four CPU threads with CUDA hidden. No teacher service, OCR, source
string, character inventory or sealed example enters the model.

Select 256 Chinese and then 256 English training filenames without replacement
from sorted admitted lists with Python `random.Random(20266900)`. Within each
source, the first 192 records are fitting data and the last 64 are calibration
data. Hyperparameters use these 128 calibration records only. Development is
all 192 existing V68 audit cases: 128 Chinese and 64 English, each with its
original matched-layout blank-question control. These are reused development
data. Do not select or exclude cases using scores.

Prepare owning prompt images and the complete first 32-by-64 reference block
on CPU and hash them before capture. The first block is a fixed physical
excerpt, not a character segmentation. The diagnostic ignores stopping and
does not expose reference length or previous/future answer blocks to the
contextual model. For each normal/control input, save the actual two
4096-dimensional initial query states, the original predicted code, continuous
pixels and stop probability. Compute detached target codes separately, after
capturing prompt states. Replay the existing development first-stop decision
and every available original first output block as a source-identity check.

## Readout intervention and elementary mathematics

Compare the unchanged original head, a two-position training-mean-code control,
a shared linear ridge map and a joint linear ridge map. The shared map treats
both query positions as samples with 4096 input and 768 output coordinates.
The joint map flattens both queries, with 8192 input and 1536 output coordinates.
Both use the same source records, target codes and hyperparameter grid.

For training-only feature mean mu and scalar centered RMS s, let X=(H-mu)/s
and Yc=Y-mean(Y). For d input coordinates solve

    min_W ||X W - Yc||_F^2 + lambda d ||W||_F^2
    A = (X X^T / d + lambda I)^(-1) Yc
    W = X^T A / d.

This is ordinary centered ridge regression, not a new learning theorem. It
removes iterative optimizer convergence as an explanation for this particular
linear fit. It neither maximizes mutual information nor tests every nonlinear
decoder. Scalar input normalization uses training data only. The elementary
spectral shrinkage factor is sigma^2/(sigma^2+lambda d) for a singular value
sigma of X; small singular modes are damped rather than presumed informative.

The fixed lambda grid is `1e-6, 1e-4, 1e-2, 1, 100`. Select separately for
each head by lowest calibration normalized-code MSE, ties by grid order. Refit
once on all 512 training records. Solve in FP64; store a standalone FP32 linear
head with training mean, scalar RMS, output mean and weights. Decode normalized
predictions through the same frozen codec in FP32. No training-example bank is
needed for head inference. The fit objective differs from V68's three-term
end-to-end objective; this is not an equal-budget causal attribution to head
capacity alone.

## Physical measurements and limits

Retain every raw code and continuous decoded image. Report normalized-code
MSE, foreground F1, exact physical blocks and blank-image fraction by source,
split, route and head. Include training/calibration before refit and all-training
fit after refit. The source-mean control receives no prompt. On development
Chinese, report normal-minus-blank F1 with 2,000 source-article bootstrap draws,
seed 20266903, and both-first-block-correct fraction on all 64 fixed pairs.
These are first-block measurements, not full-answer scores or new language gates.

Independent CPU verification reloads every source image, verifies captured-file
hashes, re-encodes targets, checks the ridge normal equations and FP32
materialization, repeats calibration selection and replays head/codec outputs.
It does not replay the entire inherited actor. Report capture, fit and verify
times separately, actual peak memory and the number of frozen-model calls.
Neither result changes V68's rejection. A successful readout would motivate a
complete causal next-block adaptation and autonomous audit; a failed readout
would leave nonlinear heads, feature learning and semantic transfer open.
