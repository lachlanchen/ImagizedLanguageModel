# Task memory influences the failed answer selector

The completed span head uses its task-memory branch, but that sensitivity
does not qualify its answers. Replacing task memory with the memory of the
other question about the same passage changes **53/64** development answer
images. The original path remains at **9/64** correct answers. The
intervention does not train or repair the model.

| Cohort | Original answers correct | Swapped answer matches own reference | Swapped answer matches partner reference | Selected image changes |
| --- | ---: | ---: | ---: | ---: |
| Previously assessed training questions | 32/32 | 5/32 | 26/32 | 27/32 |
| Development questions | 9/64 | 1/64 | 9/64 | 53/64 |

The original path has both questions correct in 16/16 training paragraph
pairs and 1/32 development pairs. After the swap, both outputs match their
respective partner references in 12/16 training and 1/32 development pairs.
All paired reference-image sets are disjoint, and both answer sets remain
physically available in each owning input.

## What changes in the conditional distribution

The sole intervention is the saved task-frame memory M. Visual features V,
source pixels, units, geometry G and all learned weights remain fixed:

\[
H(V_i,M_i,G_i)\longrightarrow H(V_i,M_j,G_i).
\]

The mean total variation between the two distributions over physical span
indices is 0.8423 for training and 0.8556 for development. This is a
span-index measure, before identical complete image candidates are merged.
The reference probability sums every admissible occurrence of that image.

The log odds of the owning reference versus the partner reference move
toward the partner in all 32 training and all 64 development interventions.
Mean owning-reference log odds change from 53.10 to -40.33 on training and
from 23.62 to -23.95 on development. These are natural-log probability ratios
of the two reference-image sets, not log odds against all incorrect answers.
Their mean can conceal individual errors.

The owning reference outranks the partner reference in 32/32 training and
45/64 development cases before intervention. The latter still gives only
9/64 correct full selections because the distribution includes many other
spans. After intervention, the partner reference outranks the owning one in
27/32 training and 47/64 development cases. Two-reference preference cannot
substitute for correct selection among all available spans.

For the first development paragraph, the original first answer selects
`为辐鳍鱼纲鲉形目鲉亚目，旧属鲉科` instead of `国公、白虎`.
With the second question's memory it instead selects that question's correct
`印度洋及西太平洋区`. Conversely, the second visual field combined with
the first memory selects `国公、白虎`. These are hybrid component calls,
not native readings of new questions. The visual field may itself contain
question information, so the intervention does not isolate a unique failure
inside the full encoder or identify a validated replacement architecture.

## Execution and scope

The [prospective protocol](contextual_query_memory_protocol.md) and producer
were committed at `57aef06` before execution. The diagnostic loads only the
completed **2,378,370-parameter** head, uses four CPU threads, and reproduces
all **96** original selected images and source boundaries. Their marginal
probabilities differ from the native GPU implementation by at most
4.293e-6; probability tensors are not claimed to replay bit-for-bit.

There are 192 head calls, zero source-encoder calls and zero optimizer
updates. Recorded invocation time is 39.644 seconds, including 1.631 seconds
setup; peak process RSS is 0.551 GiB. The separate independent verification
is excluded from that time; its duration was not instrumented. Cached features still come from the
**7,531,277,171-parameter** externally pretrained runtime. These CPU costs
exclude that encoding and do not describe a small complete language model.

All original and swapped outputs are retained. A separate NumPy/PIL
verification scans original source columns, reconstructs all 192 images,
checks the maximum-mass selected images, recomputes reference probabilities,
total variation and paired counts, and authenticates every retained file.
It verifies stored neural outputs rather than independently implementing the
head's forward pass. Large probability arrays remain under
`artifacts/contextual_query_memory_v1/probabilities/`.

The [complete raw gallery](evidence/contextual_query_memory_result/README.md)
links every original input, owning answer, swapped answer and reference.
The [machine-readable receipt](evidence/contextual_query_memory_result/result.json)
records source hashes, exact outcomes, probabilities and verification scope.
The original [failed component result](contextual_pixel_spans_result.md),
registered requirements and running adaptation comparison remain unchanged.
Public development is repeatedly reused; this is not a sealed test.

The result rejects an exact task-memory-insensitivity explanation for this
head on these cases. It does not demonstrate semantic generalization, native
instruction following, efficiency at matched language quality, or answers
absent from the input. Reliable free image-language generation remains
unachieved.

## Reproduction

Use fresh output directories and the existing Python environment; no source
model needs to be loaded for these commands:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/audit_contextual_query_memory.py \
  --output artifacts/contextual_query_memory_reproduction
python scripts/audit_contextual_query_memory.py --verify \
  --output artifacts/contextual_query_memory_reproduction
python scripts/export_contextual_query_memory.py \
  --run artifacts/contextual_query_memory_reproduction \
  --output artifacts/contextual_query_memory_public_reproduction
```

The exporter links to the existing public baseline gallery relative to the
chosen output directory. It preserves the original images and the local
probability-array identities without duplicating the baseline gallery.
