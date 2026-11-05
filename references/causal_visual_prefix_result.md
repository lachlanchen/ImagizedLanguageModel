# Causal visual-prefix audit: completed result

Executed 5 November 2026 under the
[prior plan](causal_visual_prefix_plan.md), independently of the live V63
aligned trainer. The new interface prepares pixels for a possible later actor;
it does not alter the active comparison or establish a language result.

All 12,976 complete training prompts recover exactly after blank-column
wrapping into pages. There are zero unsupported prompts and zero ink cuts.
Maximum page dimensions are 512×1,024 pixels across 16 rows. All original
whitespace and writing pixels survive; artificial row padding is removed by
the inverse. This is a complete audit of the prepared training cache, not a
claim about arbitrary scanned books or recognition under the changed layout.

On fixed CMRC and Dolly training examples, replacing every future answer
pixel with NaN leaves the visible prompt and one completed history block
unchanged. Removing all future columns also leaves them unchanged. Mutating
the original full source cannot alter the returned block, whose storage is
exactly its own 32×64 FP32 pixels. Future content is excluded from both data
assembly and pixel validation.

The real cached Qwen3-VL encoder provides a separate leakage demonstration.
The completed block comes from `cmrc/TRAIN_1000_QUERY_0.pt`; its next block and
the first target block of `dolly/dolly_05097.pt` supply the two future images.
They are concatenated after the same completed block, with identical image
dimensions and positions. Changing the future pixels changes the completed
block's first two visual vectors:

| Output | Maximum absolute feature-coordinate change |
| --- | ---: |
| Final merger | 0.901932 |
| First intermediate merger | 0.148784 |
| Second intermediate merger | 0.214888 |
| Third intermediate merger | 0.344650 |

All four independently encoded completed-block outputs replay exactly.
The demonstration uses one fixed training-image counterexample. It measures
information flow through a bidirectional image encoder, not recognition,
language accuracy or a distribution-wide effect. V63's native causal image
patch model is outside this counterexample.

The complete audit takes 47.556 seconds, with 4,515.996 MiB peak process RSS.
It runs in the unchanged native Python 3.12.1, PyTorch 2.3.0+cu121 and
Transformers 4.57.3 environment. No CUDA context, full contextual-body forward,
teacher-service call, language/image generation, development example or sealed
test is used. All frozen V63 source bytes match before and after execution.
Three interface tests pass in 1.65 seconds in the existing separate test
environment, with no skips.

The [compact receipt](evidence/causal_visual_prefix.json) pins the code, cache,
model, physical inputs and outputs. Full geometry records and feature/input
arrays are in `artifacts/causal_visual_prefix_20261105/`. The
[guide](../docs/causal-visual-prefix.md) describes the API and reproduction.
The contextual adapter, trained image writer, single-GPU resource preflight,
autonomous rollout and useful independent language remain required.
