# Conditional whole-image language model

V62 compares a single Bernoulli image head with eight conditional image
experts above the same native visual transformer. The image density is

\[
p(y\mid h)=\sum_{k=1}^{K}\pi_k(h)\prod_i\operatorname{Bernoulli}
\bigl(y_i;\sigma(a_{ki}(h))\bigr).
\]

The latent expert is shared across a complete 32x64 bitmap. This allows
correlated pixels without a character inventory. Every expert is a neural
function of context; none stores a glyph or has a fixed character identity.
The final EMA reads prompt pixels and generates new answer pixels. Typed text
is rendered before the model boundary. Source codepoints, OCR and an external
language-model call do not enter inference.

This is supervised adaptation of externally pretrained MIXAR-477M, with
attribution, rather than from-scratch single-GPU foundation training. The
single-head control has 477,145,984 parameters; eight experts have 489,998,208.
Their training data, update count and optimization are matched, but their
parameter count and output computation differ. The registered protocol is
[`correlated_native_density_v62_protocol.md`](../references/correlated_native_density_v62_protocol.md).
V61's poor free-writing result remains unchanged; a more expressive output
density is a hypothesis, not a language-quality claim.

## Reproduce the experiment

Reuse the prepared CMRC image cache and the shared pinned foundation cache.
The existing Python environment supplies Torch and Transformers. From the
repository root:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
export OMP_NUM_THREADS=4
artifacts/external/mixar_env_py312/bin/python scripts/check_correlated_native_preflight_v62.py
artifacts/external/mixar_env_py312/bin/python scripts/train_correlated_native_density_v62.py \
  --components 1 --out artifacts/correlated_native_density_v62_20261105/k1
artifacts/external/mixar_env_py312/bin/python scripts/train_correlated_native_density_v62.py \
  --components 8 --out artifacts/correlated_native_density_v62_20261105/k8
artifacts/external/mixar_env_py312/bin/python scripts/verify_correlated_native_export_v62.py
artifacts/external/mixar_env_py312/bin/python scripts/eval_correlated_native_density_v62.py
```

Run these commands sequentially. A completed result is immutable. To resume
an interrupted training arm, keep its arguments and add
`--resume <arm>/checkpoint_latest.pt`. The full checkpoint preserves the
optimizer and every RNG state. Smoke checkpoints require an explicit loader
exception and do not qualify as trained language results.

The persistent coordinator `scripts/run_correlated_native_v62.py` performs
those production stages sequentially, resumes an existing optimizer/RNG
checkpoint and verifies a complete 1,000-update report before moving to the
next arm. A zero process exit without that report is treated as interruption.
It writes actual driver/child PIDs and stage to `pipeline_status.json`, with
per-stage logs. On this workstation it runs in the single owned tmux session
`ilm-v62-training`; reuse and inspect that session rather than starting another
job. Its lock prevents a second coordinator for the same run. SIGINT/SIGTERM
are forwarded to the active training child so it can save its state.

The preflight checks four real updates against interrupted/resumed training
for each arm, full-backbone gradients, future-pixel causality and cached
inference. TF32 can produce shape-dependent logit differences between long
prefill and one-block decoding; cache semantics are checked separately with
strict FP32, with both discrepancies retained. Production precision remains
the registered setting.

## Use learned weights

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python scripts/infer_correlated_native_language.py \
  --checkpoint artifacts/correlated_native_density_v62_20261105/k8/model_final.pt \
  --prompt-image prompt.png --out answer.png --export deployment_bundle
```

Input PNGs are aligned 32-pixel-high writing strips, up to 480 cells. The
output includes the actual binary image tensor, a PNG and decoder events.
An empty answer has an empty tensor and a white display placeholder; that
placeholder is not scored as generated writing. Each expert's complete mode
is scored under the full mixture. This candidate search is approximate MAP,
not a guaranteed global mode. The selected physical pixels are fed back;
no malformed stroke is repaired or snapped to a font.

The deployment directory contains learned weights, local architecture code,
foundation attribution and hashes. Its `run.py` takes a prompt PNG without
the repository or the original source weights:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
artifacts/external/mixar_env_py312/bin/python deployment_bundle/run.py \
  --prompt-image prompt.png --out answer.png
```

Offline replay and the locked question audit are separate checks. Exact replay
establishes reproducibility; it does not establish correct answers. The audit
retains normal/blank-question controls, both-correct question pairs, raw
rasters, likelihood, expert-use diagnostics and measured GPU resources.
