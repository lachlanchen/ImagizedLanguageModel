# Trainable readout for continuous image-language generation

The native-context model fails to predict clean answer images even when given
correct answer history. The [completed diagnosis](../references/native_corpus_fit_v80_result.md)
gives 50.241% foreground F1 under that control, compared with 99.9953% from
supplied-code reconstruction. The next experiment tests a trainable readout
and transition over the same full prompt memory.

The [implementation](../ilm/visual_lm/adaptive_image_continuation.py) adds one
continuous initial read query and reuses rank-8 adapters on q/v/o projections
in all 36 inherited layers. Adapters affect writing queries only. Actual prompt
prefill uses the unchanged base weights, allowing its KV cache to remain valid
through optimization. Each emitted block is decoded to pixels and re-encoded
before continuing. Inference adds no tokenizer, language output vocabulary,
OCR, teacher service or glyph lookup table.

The adapted actor has 7,583,225,971 complete parameters, of which 53,096,961
are trained. The matched control retains 7,583,221,875 complete parameters and
46,899,713 trainable parameters. Both inherit external pretraining. These
counts describe an adaptation experiment, not a lightweight model trained
from scratch.

The [real-model preflight receipt](../references/evidence/adaptive_readout_v81r2_preflight.json)
records finite nonzero gradients in all query, adapter and image-interface
groups; exact model/Adam/noise resume; unchanged frozen parameters and prompt
KV after adapter updates; causal future exclusion; and actual emitted-pixel
feedback replay. Peak allocated memory is 18.613 GiB on GPU0. The original
absolute numerical tolerance fails for the adapted path. Its error and exact
replay are preserved; the [explicit numerical revision](../references/adaptive_readout_v81r2_protocol.md)
uses a vector-scale comparison and still reports the original failed absolute
check. This preflight demonstrates execution, not language quality.

Run the matched experiment in the existing compatible environment:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/train_adaptive_readout_v81.py --stage train
```

The unchanged control and adapted arm run sequentially for 2,000 updates each,
using the same 32 fixed training questions, fresh Adam states, case order and
noise draws. Both start from the same failed native checkpoint. The query and
transition changes form one composite intervention. Initial and final raw
answers for every case are retained; final correct-history and supplied-code
controls are retained too. No best-checkpoint selection or early stopping uses
these answers. Runtime files stay in
`artifacts/adaptive_readout_v81r2_20261106/`.

After both arms finish, independently verify on CPU:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python \
  scripts/train_adaptive_readout_v81.py --stage verify
```

The provisional training-fit requirement is at least 24/32 exact free answer
images and foreground F1 at least 0.98, with an improvement over the matched
control. No experimental training-fit result is available at the time of this
preflight record. Passing that requirement would still leave new-prompt,
copying and development evaluation necessary. A usable general ILM and an
efficiency advantage over conventional language models remain unestablished.
