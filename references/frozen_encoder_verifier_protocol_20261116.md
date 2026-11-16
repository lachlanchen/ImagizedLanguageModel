# Fixed learned-feature metric for complete-answer qualification

The original training-only Hamming verifier rejects two known-correct answers
and accepts one uncertain answer on the complete 64-case cohort. Its bank has
9,581 training-source forms. The subsequent full source audit verifies every
cell and shows why physical identity must be kept separate from raw Unicode.
The new candidate changes **only the distance representation**: use the existing
frozen glyph-image encoder in place of bit-space distance. It fits no new model
and does not resume either rejected general-vision teacher.

## Bound representation

Use only the encoder tensors from the original balanced bridge checkpoint,
SHA-256 `dcac4ee207614adca78a8b99cc4fe422b3197204d9028192bef30f111d61586d`.
Its preceding glyph primitive was trained with image/semantic alignment,
reconstruction and one-pixel shifts; those tasks do not establish language
or recognition of arbitrary emitted forms. The old reconstruction and language
results remain unchanged. Load no 7.5B native body, lexical table, renderer or
new teacher. Bind the checkpoint, exact encoder architecture, inherited matcher,
full data sources, new implementation, tests and protocol before neural calls.

Feed complete 32×32 cells as +1 white and −1 ink into the eval-mode frozen
encoder. Use its raw 4,096-dimensional normalized-semantic coordinates, without
the global feedback affine, a new projection, feature normalization, augmentation,
fine-tuning or label fitting. Encode in fixed batches of 64, padding the last
batch with white images and removing padding afterward. Disable TF32 and use
deterministic CUDA/cuDNN. Forward arithmetic is FP32. Compute direct Euclidean
distances in FP64, without the norm-minus-dot-product identity.

Construct the same complete physical training bank as the prior verifier,
using all original training shards and no development/reference additions.
Unicode labels do not enter it. For prototype i let delta(i) be its nearest
other prototype's feature distance. Accept a query's nearest identity only
when twice its distance is strictly below delta(i). Ties, zero separations and
outside-ball queries abstain. No radius is tuned. Disjoint feature-space balls
give unique decoding under bounded **feature** corruption; they do not certify
human legibility or a bounded visible-pixel perturbation. Learned invariance is
a hypothesis to test, not a presumed benefit over Hamming distance.

## Unchanged complete cohort and decision

Use all 64 frozen final normal QA outputs: 9 strict correct, 9 readable wrong
and 46 uncertain/uncredited. Preserve their images, transcriptions and uncertainty.
No old output becomes training data or selects a metric, checkpoint or threshold.
The decoder sees only cell pixels and the training bank; questions, references,
language context and source IDs cannot influence its decisions. The inherited
separate matcher compares the ordered physical identities with complete
reference images, retaining duplicates and spaces. The catalogue exists only
for this training/evaluation component, never deployed answer generation.

Keep the original score: half the ordered identity-edit similarity plus half
the exact complete-identity indicator, with cap/all-white/empty reward zero.
Require all clean references covered, all nine correct answers fully accepted,
no readable-wrong or uncertain complete answer accepted, correct/wrong pairwise
AUC at least .95, every correct reward strictly above every uncredited reward,
every cap zero and all prototype feature separations positive. Uncertain old
readings are not proven semantic errors; preserve them when the conservative
qualification rejects a judge. No independent human adjudication is added.

Any failed condition rejects this exact feature map/metric. Do not try another
normalization, layer, checkpoint or radius on these outputs. A pass qualifies
only this reused-output component and still requires a separately registered
fresh-clone actor own-image sampling, full-action replay, gradient and resource
preflight. No student optimization or language gain follows calibration alone.

## Resource and provenance

One serial process owns the project GPU0 mutex and pins CUDA to the physical
GPU0 UUID before importing Torch. Record private PID/start time/boot identity.
Require 48 GiB available host RAM and 20 GiB free GPU0 at entry, retain 24 GiB
host RAM, and cap process RSS and allocated GPU memory at 2 GiB each. The whole
action and inherited measured calibration each have a 300-second ceiling. Use
two CPU threads and the fixed CUDA workspace. No service, weight download or
foreign-process shutdown. Foreign GPU fluctuations are not attributed to this
UUID-pinned process. Preserve partial files and stop on a failure; no retry.

Report encoder parameter count, all forward calls, actual encoded positions,
feature/radius hashes, package versions, stage and whole-action time, peak RSS
and VRAM. These are component costs, not independent-model or matched-quality
efficiency evidence. All original source bindings and failed rewards remain.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python -m unittest tests.test_frozen_encoder_verifier -v
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/calibrate_frozen_encoder_verifier.py register
# Commit and push first, then run once.
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/calibrate_frozen_encoder_verifier.py run
```
