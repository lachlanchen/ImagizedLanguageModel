# Most retained writing steps have no matched text-prefix state

The retained source-helper and image-writer traces support only **2 of 75
intermediate history correspondences** on the fixed 32 training questions.
The initial states agree in all 32 cases. This audit does not train or repair
the writer; it prevents an invalid comparison between histories with
different visible content.

| History position | Image boundaries | Exact canonical text-prefix raster exists | Matching saved source state exists |
| --- | ---: | ---: | ---: |
| Initial | 32 | 32 | 32 |
| Intermediate | 75 | 7 | 2 |
| Terminal | 32 | 32 | 19 |

At **68/75 intermediate boundaries**, no complete canonical reference-text
prefix produces the same raster prefix under the fixed rendering convention.
At five other intermediate boundaries, a suitable raster prefix exists but
the saved helper trace has no corresponding state. All terminal reference
rasters are representable, while thirteen lack a matching helper state.
Different helper answers and missing matching prefixes remain in these
denominators; no cases are discarded based on their outcomes.

All **216** eligible saved source-prefix positions in this cohort decode to
complete UTF-8. Split-byte prefixes therefore do not explain the missing
correspondences here. Native language pieces and physical image blocks still
have different boundaries and step counts.

## What was compared

The [earlier state-alignment experiment](native_state_alignment_protocol.md)
supervised Show-o2's pre-answer prompt states. It did not test alignment of
the Qwen writer's new image-history transitions. The present audit joins the
existing [source-helper traces](reader_content_v77_result.md) to the
[correct-history writer traces](native_corpus_fit_v80_result.md), using all
32 questions shared by their authenticated training cohort.

Owning physical-prompt hashes and BF16 reader-memory hashes agree for every
case. Initial source and writer states also agree after common BF16 rounding.
The writer then consumes two learned image embeddings per 64 answer columns;
the helper consumes native language pieces. A shared ordinal index does not
identify a shared prefix.

For each original image boundary, the audit tests every complete reference
text prefix against the actual preceding answer pixels. Prefixes retain the
full original reference's font, canvas and draw origin. The renderer normally
centres each string from its ink bounds, so independently rendering each
short prefix would introduce an additional vertical-layout change. Keeping
the original origin avoids that confound. Prefix shaping still uses the
original font renderer; the audit does not independently implement shaping
or claim a universal character segmentation rule.

A saved helper state is eligible only when its decoded preceding text is one
of those physically matching prefixes. All eligible states are retained.
These are the two intermediate matches:

| Training question | Visible prefix | Image blocks consumed | Image input positions | Source language positions consumed |
| --- | --- | ---: | ---: | ---: |
| `TRAIN_1779_QUERY_1` | 美国维吉尼亚州的 | 3 | 6 | 6 |
| `TRAIN_2146_QUERY_1` | 1.7-2. | 1 | 2 | 6 |

Their normalized-state RMSE values are 1.497 and 1.407. The first has equal
input-position counts; the second does not. These vector distances are not
calibrated language scores. No vocabulary projection of the image-history
states or controlled alignment-learning intervention was performed. The
measurements do not identify history alignment as the cause of the corrupted
answers or establish it as a repair.

## Reproducibility and consequence

The [protocol](history_prefix_alignment_protocol.md) and producer were
committed at `bfa8026` before execution. The CPU audit takes 13.270 seconds
and peaks at approximately 1.000 GiB process RSS. It reads saved states and
read-only tokenizer-string metadata, constructs no neural model, performs
zero model forwards and makes zero optimizer updates. No source weights are
dequantized or copied. The tokenizer string table is hashed; the complete
shared archive is explicitly not rehashed in this metadata-only operation.
The decoder is an offline instrument, with no change to deployed inference.

An independent NumPy/PIL check reconstructs UTF-8 prefixes from the saved
pieces, checks initial BF16 rounding, rescans every physical correspondence
and recomputes matched-state distances and all totals. It verifies retained
arrays rather than rerunning either contextual model or the font renderer.
The [complete machine-readable receipt](evidence/history_prefix_alignment_result.json)
contains all 32 cases, source identities, matches and verification scope.
The 5.18 MB of prefix PNGs and state arrays remain in
`artifacts/history_prefix_alignment_v1/` with hashes in that receipt.

These traces cannot support a general pointwise comparison or a complete
teacher-state training set at the existing 64-column boundaries. A future
experiment must explicitly define how teacher and actor histories correspond,
for example by using compatible emission boundaries or a teacher that
observes the same partial raster history. Those are possible experiments,
not selected or validated solutions. Future-conditioned labels can be used
as predictive supervision, but must not be described as states of an
identical causal history or supplied to autonomous inference.

The running contextual adaptation comparison is unchanged. This training-set
audit adds no natural-answer score, free image output, efficiency comparison
or historical-script capability. The independent ILM remains unqualified.

Reproduction uses a fresh output directory and the existing renderer:

```bash
LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libglib-2.0.so:/usr/lib/x86_64-linux-gnu/libgobject-2.0.so:/usr/lib/x86_64-linux-gnu/libgio-2.0.so' \
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python \
  scripts/audit_history_prefix_alignment.py \
  --output artifacts/history_prefix_alignment_reproduction
python scripts/audit_history_prefix_alignment.py --verify \
  --output artifacts/history_prefix_alignment_reproduction
```

CMRC2018: Cui et al. (2019), CC-BY-SA-4.0. Qwen3-VL: Alibaba, Apache-2.0.
