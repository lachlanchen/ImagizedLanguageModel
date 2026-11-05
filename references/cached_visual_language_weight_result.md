# Legacy visual-language weight inspection: completed result

Executed 5 November 2026 under the
[prior CPU-only plan](cached_visual_language_weight_plan.md). V63's aligned
trainer continues independently on GPU 0. No architecture or quality gate in
that comparison changes.

The existing Qwen3-VL 8B combined GGUF is authenticated against manifest SHA
`0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319`
and full blob SHA
`1329cd5ab37e5ef6faef37cd8ff8660ffc96ed5b47b34fddf8eb0912e6c82466`.
The independent accessor retains 397 contextual and 351 native vision
parameters. Every name and shape matches the installed Transformers 4.57.3
schema, and the meta contextual embedding has zero vocabulary rows. The full
contextual body is not materialized or executed.

All 576,388,336 vision parameters are dequantized, checked finite and streamed
into one CPU encoder. Fixed training-image crops come from
`cmrc/TRAIN_1000_QUERY_0.pt` at x=4160 and `dolly/dolly_05097.pt` at x=2496.
Each crop is 32×64 pixels from the middle physical block of its prompt; these
are diagnostic crops, not complete-page reading tests. The image-only
processor packs them as grid `[1, 2, 4]`, with no resize or text processor.
Both produce finite `(2, 4096)` final features and three intermediate feature
sets of the same shape. Repeating the second input reproduces all four outputs
exactly. No semantic correctness or unfamiliar-script capability is measured.

Independent float64 contractions on the actual source projections give:

| Storage operation | Maximum absolute numerical difference |
| --- | ---: |
| Flattened patch kernel versus explicit channel/time contraction | 7.21645×10⁻¹⁶ |
| First visual layer fused versus separate Q/K/V projections | 0 |
| Last visual layer fused versus separate Q/K/V projections | 0 |

The entire inspection takes 28.886 seconds including full blob authentication,
with 4,534.637 MiB peak process RSS. It uses no CUDA context, model service,
language decoder, tokenizer or image-generation call. All eight frozen V63
source hashes match before and after execution. The model weights are reused
in place; no model weights or SDKs are downloaded, copied or installed.
The one downloaded 16,049-byte upstream Go source is a pinned layout reference,
not a model or an executed dependency.

Three tests pass on the actual archive: native meta-schema compatibility,
independent real-weight contractions, and rejected vocabulary/unattributed
position configuration. None skips. They execute in the existing separate
test environment; the completed native encoder run uses the unchanged
Python 3.12.1, PyTorch 2.3.0+cu121 and Transformers 4.57.3 runtime.

The [compact receipt](evidence/cached_visual_language_weights.json) pins the
plan, implementation, test source, inputs and outputs. Full report SHA is
`22deb4394b2e4593d487d00d7b22b69b430e20c856d7c1a8bb9285897d2bdfef`.
The [guide](../docs/cached-visual-language.md) gives reproduction commands.

This completes storage compatibility and bounded real vision execution.
The retained 7.522B source parameters would occupy 14.012 GiB as dense BF16
weights alone. No GPU-training fit, complete image-language actor, full-body
forward equivalence or language-quality result follows from that calculation.
The independent language goal remains active and requires actual autonomous
image answers.
