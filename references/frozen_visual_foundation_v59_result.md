# Frozen visual foundations: V59 result

The registered frozen-feature comparison is negative. A fresh image-block
readout on MIXAR-477M features answers 6/256 questions exactly (2.344%) with
11.588% multiset raster F1. The identical-size readout on V58 student features
answers 24/256 (9.375%), F1 25.340%. MIXAR-minus-student F1 is −13.752 percentage
points, source-article 95% interval [−18.247, −9.094]; exact match is −7.031
points [−10.646, −3.101]. Both questions are correct in 0/393 MIXAR pairs,
versus 12/393 student pairs. The probe passes three of seven registered gates
and its decision is **reject_frozen_foundation_probe**.

Blank-question F1 is 9.341% for MIXAR and 6.997% for the student. The respective
normal-minus-blank differences are 2.246 [0.415, 4.378] and 18.343
[13.949, 22.519] points. A small question effect does not rescue poor absolute
answer quality. No reference or answer offset enters either feature encoder.

All 3,211 train and 1,321 development prompts, plus 1,321 blank-question
prompts, are cached as image-only continuous features. Both encoders receive
identical binary Noto CJK cells. Pair-completion alignment, fixed projections,
fine DCT features, phase channels and FP16 quantization are registered. The
514-to-384 readouts each train 789,520 parameters for 4,000 matched updates
and batch eight. The student's source encoder already has task training;
MIXAR has different pretraining, capacity and native rendering. This is not
a controlled comparison of pretraining or a native-font MIXAR competence test.

Feature preparation takes 184.781 s at 1.906 GiB allocated GPU memory.
Student and MIXAR readout training take 106.593 and 109.257 s, each peaking at
0.170 GiB. The cached readout audit takes 117.085 s, 0.094 GiB. All local GPU
phases run sequentially on physical GPU 0, an RTX 4090 D. Cached audit costs
exclude encoder inference; the author's reported external four-H200 pretraining
is not single-4090 work. The pinned Apache-2.0 release and its weights are
reused from the shared cache, not duplicated.

GPU resume preflight reproduces all model/EMA weights and RNG states exactly;
cached feature recomputation has zero difference after FP16 quantization.
Both completed EMA readouts precede development inspection. The same 256 IDs
and 393 pairs are explicitly reused development. No independent final test,
free writer or new historical-script capability is qualified by this probe.

Files are in `artifacts/frozen_visual_foundation_v59_20261105/`, with `features/`,
`student/`, `mixar/` and immutable `development/` audit/raw answer images.
[Protocol](frozen_visual_foundation_v59_protocol.md),
[feature cache](../scripts/cache_frozen_visual_features_v59.py),
[training](../scripts/train_frozen_visual_readout_v59.py),
[audit](../scripts/eval_frozen_visual_readout_v59.py).
