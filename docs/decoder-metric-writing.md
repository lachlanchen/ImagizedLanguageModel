# Decoder-sensitive continuous writing experiment

The completed readout repair still fails autonomous writing. This experiment
tests one training change: weight continuous-code errors by the local
sensitivity of the frozen glyph decoder. The autonomous model, continuous
query, suffix adapters, sampler and actual-pixel feedback are unchanged.
No tokenizer, vocabulary, OCR, lookup or teacher service is added at runtime.

The [loss implementation](../ilm/visual_lm/decoder_pullback.py) computes
`mean(v²) + 0.001 mean((J_R(x)v)²)` using a Jacobian-vector product through
the normalized decoder. The metric point is the observed noisy code, detached
from optimization. Positive code MSE keeps the metric positive definite and
preserves the conditional-mean optimum under the stated state-only assumptions.
This is an optimization hypothesis, not a proof of language learning.

[Actual decoder derivative checks](../references/decoder_pullback_v82_preflight_result.md)
pass on CPU and GPU0. The [complete-actor preflight](../references/evidence/decoder_metric_v82r2_preflight.json)
also passes: finite nonzero gradients in every trained group; exact optimizer,
sampler and noise resume; unchanged frozen parameters and all prompt KV;
exact future exclusion; actual generated-pixel feedback replay; and 18.615 GiB
peak allocated VRAM. The strength-zero objective matches the original loss
on an actual unpadded example.

The first actor preflight fails its padded-batch numerical criterion and is
preserved. The [execution revision](../references/decoder_metric_v82r2_protocol.md)
computes each owning contextual history at its true length in both arms,
then pads the resulting states for the shared writing loss. This makes
padding comparisons bit-exact; both arms also pass the original absolute
1e-4 incremental check. Numerical and quality thresholds are unchanged.
This execution choice may cost throughput and is not an efficiency result.

Both arms restore the same completed V81 adapted checkpoint and receive the
same case order and noise draws for 2,000 additional updates each. Complete
parameter count is 7,583,225,971, including an externally pretrained foundation;
53,096,961 parameters are trained. The control uses strength zero; the metric
arm uses 0.001. All 32 questions are seen training cases. No autonomous
quality result is available at this preflight stage.

Run from the repository root in the existing compatible environment:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/train_decoder_metric_v82.py --stage preflight
```

After the actual preflight passes, use the same environment with `--stage
train`. The currently running invocation already owns GPU0; do not launch a
duplicate. Artifacts are under `artifacts/decoder_metric_v82r2_20261107`.
Completed earlier runs and the original failed preflight remain separate.
Source identities protect checkpoints and partial outputs from mismatched
resumes. Do not edit the pinned implementation during a run.

After both arms complete, independently score and re-decode on CPU:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python \
  scripts/train_decoder_metric_v82.py --stage verify
```

The original training-fit requirements remain at least 24/32 exact free
answers and foreground F1 at least 0.98, with both exceeding the matched
control. All before/after outputs are retained, and all 64 initial streams
must exactly replay the original warm model. A passing result would still
require new-prompt/copy and development evaluation. General language ability,
historical-script transfer and efficiency parity remain unestablished.

## Result export and checkpoint inference

After both arms and the CPU verifier finish, validate and export every saved
answer with CUDA hidden:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python \
  scripts/export_decoder_metric_v82_result.py --check-only

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python \
  scripts/export_decoder_metric_v82_result.py
```

The exporter checks both complete checkpoints and rescoring, retains the
original four training-fit decisions and copies every before/after PNG
unchanged. Its plates show all cases in order, including wrong and empty
answers. The unfinished-run rejection was exercised against the actual live
experiment without creating an evidence directory. The successful export
path remains pending until that experiment completes.

The [checkpoint loader](../ilm/visual_lm/metric_image_actor.py) builds the
inference actor from the final learned checkpoint, shared frozen foundation,
writing codec and numeric task controls. It records the warm-checkpoint hash
as ancestry without opening the warm file or loading a training dataset.
It accepts the declared completed V82 checkpoint format; completion metadata
does not establish language quality.

The [CLI](../scripts/infer_decoder_metric_v82.py) accepts a prompt PNG,
optional host-rendered text/text file, or an instruction above an image. An
example input-only preparation is:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python scripts/infer_decoder_metric_v82.py \
  --checkpoint artifacts/decoder_metric_v82r2_20261107/metric/checkpoint_latest.pt \
  --text '请用中文解释汉字“言”的本义。' \
  --prepare-only --output artifacts/metric_prompt_preparation
```

Four actual CPU preparations pass: typed text, a UTF-8 text file with explicit
newlines, an image, and instruction-plus-image. Tensor/PNG equality and the
unchanged image region in the mixed prompt are verified. Incomplete and
incompatible real checkpoints are rejected before CUDA initialization. See
the [preparation receipt](../references/evidence/decoder_metric_v82_runtime_preparation.json).
These are input and guard checks. Full restored-model generation and an
independent checkpoint replay remain pending; no new language result follows.

After the current GPU job finishes, the same CLI performs research inference
without `--prepare-only`, using the deterministic GPU environment above and
a completed checkpoint. It saves raw answer tensors, PNGs and provenance.
Its image output is unmodified, and no transcription or answer correction is
added. Use a fresh output directory for each invocation.
