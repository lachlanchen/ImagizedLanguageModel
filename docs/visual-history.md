# Reproduce the auxiliary visual-history studies

The [results](../references/visual_history_reading_result.md) do not qualify
an image-answer model. The first study uses an offline text projection to
measure a frozen source reader; the second learns numeric state maps and
assesses single native pieces. Neither generates ILM answer images.

Use the existing shared Qwen3-VL archive and Python environment. Do not
duplicate weights or start another project job while either stage is active.
The original local runs are `artifacts/visual_history_reading_v1` and
`artifacts/visual_history_bridge_v1`. The preserved raw states and fold models
are local; public evidence retains all responses and input PNGs.

Prepare a fresh native-reading run on CPU. The renderer requires the existing
system GLib preload:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4 \
LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libglib-2.0.so:/usr/lib/x86_64-linux-gnu/libgobject-2.0.so:/usr/lib/x86_64-linux-gnu/libgio-2.0.so' \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
scripts/probe_visual_history_reading.py --stage prepare \
--output artifacts/visual_history_reading_reproduction
```

Inspect the actual `inputs/` pages before generation. Check host memory,
project processes and GPU0 occupancy. Generate all 388 fixed calls under the
single project GPU lock. A resumed invocation authenticates completed calls
and performs only missing calls; a completed run is immutable.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 \
MKL_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$HOME/.local/bin/with-current-nvidia" \
artifacts/external/mixar_env_py312/bin/python scripts/probe_visual_history_reading.py \
--stage generate --output artifacts/visual_history_reading_reproduction
```

After the source process exits, verify on CPU with the same GLib preload as
preparation, changing `--stage prepare` above to `--stage verify`. This checks
page construction, control compilation and all native projection decisions
from the retained states. It does not repeat full vision/body inference.

The analytic learner uses a separate fresh directory. Every fold withholds
an entire source pair and all its prefixes. No remaining-answer string enters
fitting; its original target text states are privileged teacher data.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
scripts/fit_visual_history_bridge.py \
--source artifacts/visual_history_reading_reproduction \
--output artifacts/visual_history_bridge_reproduction
```

Independently refit/check all folds with NumPy and reconstruct their reports:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
scripts/export_visual_history_evidence.py --stage verify-bridge \
--source artifacts/visual_history_reading_reproduction \
--bridge artifacts/visual_history_bridge_reproduction
```

The registered identity, mean and ridge controls all remain. The production
ridge fit contains 22 folds; three earlier partial fits are explicitly
reported after a numerical-check correction. The original failed probe gate
and the separate failed alignment gate remain unchanged. Do not deploy a
cross-fitted map or temporary text projection as a qualified image chatbot.

The separate [cursor history](../references/cursor_history_result.md) repairs
a whitespace-state collision in a future image-only feedback interface. It
is not present in the source probe above. Before adopting it in a learner,
encode and assess its actual pages; cached states from the old layout do not
become valid for a new image merely because its visible wording is similar.
