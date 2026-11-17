# Fixed 256-bit scalar-plan comparison

Use all 68,948 original source images and the unchanged exact-mask partition:
64,041 training cells / 8,629 forms and 4,907 validation cells / 952 forms.
The empty mask remains training-only; every physical alias stays in one split.
Keep uniform-form representative sampling and all original source occurrences
in evaluation. The reused physical split is development, not untouched language
confirmation. No reserved DRCD or other language records are rendered or scored.

Change the 256-bit plan's allocation from sixteen one-bit features to eight
two-bit finite scalar features per 4-by-4 site, as defined in the
[design](scalar_visual_plan_direction_20261118.md). The new paired control uses
128 continuous values bounded identically before rounding. Both modes share
exact initial tensors and have 1,768,393 parameters; verify all 1,766,209 common
CNN initial values against saved spatial-256 controls. Those controls are not
retrained, redecoded or selected from intermediate checkpoints.

Keep seed 202611195, order seed 202611196, 6,000 updates per arm, batch 128,
768,000 retained draws per arm, FP32, deterministic kernels, no TF32, AdamW
(.9,.95), weight decay .01, clip 1, 250-update warmup to .0002 and cosine decay
to .00002. Retain the original per-image boundary BCE plus .5 Sobel and .5 Dice
objective. Before fitting each arm, differentiate every training image with
equal-form source weights, perform one real update, measure and reset exactly.
Verify an exact model/optimizer replay at update three and record its extra
128 trial-image cost. No curriculum, auxiliary objective or extra exposure.

Evaluate initial and final models on all source images in original row order.
Store actual pixels, all 256-bit plans or 128 continuous values and all actual
XOR-tree event counts. Checkpoints explicitly record architecture, eight planes,
four levels, two bits per scalar, 128 scalars and 256 total bits. The historical
mode name `binary` denotes serialized hard bits, not one-bit scalar features.

Preserve every original absolute criterion: quantized training and validation
pooled F1 at least .98, validation macro-form F1 .98, macro exactness .80,
continuous validation ceiling .98 and quantized retention within .005. No source
false blanks or false ink. Thirty-two plan events plus actual correction events
must average at most .60 of 97.30248883216336 on the complete source. Additionally
require validation F1 gain .02 over saved binary .9436900709752929. Report all
failures; the continuous exact rate is also reported without adding a new gate.
Event counts are neither compressed bytes nor throughput.

Use one serial owner under artifacts/ilm_gpu0.lock. Fresh admission requires
48 GiB available host, 20 GiB free GPU0, 48 GiB disk and low memory pressure.
Runtime retains 24 GiB host and ceilings of 8 GiB allocated GPU, 8 GiB process
RSS, two CPU threads and 2,400 seconds including at most 300 seconds pressure
wait. External timeout 2,460 seconds plus 30 seconds grace. Do not touch foreign
services. Preserve every failed execution; no automatic retry.

Only final checkpoints qualify. A failure ends this recipe; preserve all images,
ledger rows, hashes and limited positive effects. A pass permits separately
registering conditional-image learning, not autonomous language or deployment.
The whole-model language, stopping, independence, historical-form and cost
criteria remain unchanged.
