# Visual features and context both affect Chinese copying

The fixed-weight, 128-call comparison does not find a working reading repair.
Truncating the square image's visual features to the original short context
still gives **0/16 exact Chinese copies**. Substituting the original strip's
features into the long context recovers **16/16 English copies**, but only
**2/16 Chinese copies**. These results narrow the previous padding diagnosis;
they do not qualify language understanding or image answers.

Let S denote the 1,024 visual features from an original 1024-by-256 strip.
Q denotes the 4,096 features from its 1024-square version, formed by adding
only white bottom padding. The first 1,024 Q features cover the same upper
256 pixels. Every request's ink lies inside that strip. No source pixels are
resized, removed or moved.

| Arm | Visual feature block | Prefix positions | Chinese exact stopped /16 | English exact stopped /16 |
| --- | --- | ---: | ---: | ---: |
| strip_short | S | 1,063 | 13 | 16 |
| square_long | Q | 4,135 | 0 | 7 |
| square_short | First 1,024 Q features | 1,063 | 0 | 6 |
| strip_long | S, then Q's remaining 3,072 features | 4,135 | 2 | 16 |

All **64 original strip and square decision sequences replay exactly before
the 64 hybrid calls**. The nonvisual header and suffix are bit-identical
between each pair of source prefixes. Native source masks, model weights,
generic instruction, row order, seeds, top-one readout and 96-decision/EOS
limit remain fixed. All 128 unmodified responses are retained in the
[complete export](evidence/native_canvas_factorization_result/result.json),
bound to the source PNGs in the preceding
[canvas experiment](evidence/native_canvas_control_result/result.json).

The first English reference is `garden silver open blue`. Its source strip
response is correct; the square gives `green silver silver`; the square's
top features in the short context give `green silver orange black`.
Substituting the strip features into the long context restores the correct
answer. For the first Chinese reference, `树田红星风`, the same four responses
are `树田红星风`, `林石三石林`, `山南云里里山南云` and `树红红色风`.
These are content changes, not harmless formatting differences.

The complete invocation takes **68.220 seconds** after model/codec loading,
with **11.161 GiB** peak allocated VRAM on one RTX 4090 D. It loads the fixed
2,839,267,040-parameter dense source and 126,892,531-parameter codec; all
parameters are frozen and there are zero optimizer updates. No model weights
are downloaded or duplicated for the comparison. Source checkpoint SHA256:
`016429ec5454cdd681693d881df4c6b7d52fb43dcae7b9e15857d2d9cd32b483`.

## What this establishes

Shortening the square context alone does not restore copying. Restoring strip
features recovers English copying even in the long context. Chinese still
degrades substantially relative to its strip source, so preserving that
feature block alone is also insufficient in the tested long-context layout.
A geometry-invariant frontend or a context repair must be demonstrated;
neither has been established by the previous attention-only training run.

The hybrids are representation interventions and need not encode any real
image. Nominally blank-region Q features may contain global information.
Changing context also changes positions, suffix placement and available keys.
Equal seeds across different tensor shapes do not imply identical noise.
These controls therefore do not uniquely attribute failure to VAE sampling,
positional interpolation, visual attention, language attention or another
individual operation. They do not establish that information is irrecoverably
lost in Q. No natural QA or image generation is assessed in this comparison.

The separate [attention learner](native_reader_attention_result.md) still
has 0/32 correct Chinese and 0/32 correct English QA answers after 2,560
updates. It preserves copying on its calibrated strips; that is a prerequisite,
not semantic learning. The unchanged recipe is not admitted for book-scale
training. The independent image-input/image-output ILM remains unachieved.

## Reproduction

The [prospective protocol](native_canvas_factorization_protocol.md) fixes
all arms before neural execution. Reuse the existing pinned runtime and source
checkpoint; run one project GPU job at a time:

```bash
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  artifacts/external/native_unified_source/runtime/bin/python \
  scripts/probe_native_canvas_factorization.py --stage evaluate \
  --output artifacts/native_canvas_factorization/assessment_v1

env CUDA_VISIBLE_DEVICES= PYTHONPATH=. \
  artifacts/external/native_unified_source/runtime/bin/python \
  scripts/probe_native_canvas_factorization.py --stage score \
  --output references/evidence/native_canvas_factorization_result
```

Both output directories are created exclusively; a completed or incomplete
run is preserved rather than silently restarted. Choose fresh output paths
for a new invocation and pass its evaluation path to scoring with `--run`.
The scorer authenticates all source files, outputs, source replays, input
hashes, seeds, prefix lengths, nonvisual checks and implementation bindings.
Copying references are loaded only for offline scoring.

The requests are synthetic. Show-o2 implementation and weights are NUS Show
Lab, Apache-2.0. This auxiliary Unicode-head diagnosis does not constitute
an independent image-output language model.
