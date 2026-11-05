# Native visual-question capability and a reader-scale mismatch

The authenticated source VLM answers 13/16 fixed Chinese image questions
exactly with a generic task instruction, versus 0/16 with question ink erased.
Five of eight source-article pairs have both answers exact. All 96 requests,
48 source PNGs and literal-reference scores pass independent verification.
This establishes a bounded capability of the external source through its
native interface. It is not an image-language result of the ILM.

## Fixed native-source comparison

The source model is the same authenticated Qwen3-VL 8B Q4 archive used for
initialization. Sixteen Chinese questions are the first eight article pairs
from the existing V68 audit. Eight English instructions are the first
lexicographic case in each category. These are reused development cases;
selection precedes source-model calls. None may become a training target.

| Native route | Chinese exact / 16 | Chinese character accuracy | Both exact pairs / 8 | English exact / 8 | English valid JSON / 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image + fixed task instruction | 13 | 90.916% | 5 | 1 | 8 |
| Image, empty text content | 10 | 76.404% | 3 | 1 | 6 |
| Blank-question image + instruction | 0 | 2.218% | 0 | 0 | 8 |
| Matched source text + instruction | 10 | 85.442% | 3 | 1 | 8 |

All 64 Chinese responses have valid JSON and completed termination. Two English
image-only responses have invalid schema and receive no credit. The scorer
trims outer whitespace only and accepts all retained reference strings.
Character accuracy uses bounded normalized edit distance. English results
are literal reference agreement, not exhaustive semantic correctness.

Literal agreement also understates some Chinese answers. On the first case,
the image route emits `玛丽莲·曼森`; the image-only and text routes emit
`Marilyn Manson`, which does not equal the retained Chinese reference. These
counts do not establish that image input is superior to text, or estimate
population performance from 16 questions.

The image routes receive complete original actor pages as RGB PNGs, with no
client-side resize or crop. The source API receives no case identifiers,
references or score metadata. Native chat structure and the answer schema
still scaffold the image-only route. All source outputs use its pretrained
tokenized text decoder; no such decoder or service is added to the ILM.

## Observed preprocessing difference

The owned source runtime actually launches with `--image-min-tokens 1024`.
Its log records a 1,048,576-pixel minimum and 4,194,304-pixel maximum. The
current custom reader processes the original pages without that resize.

For the first 224-by-1024 page, the custom reader has 224 merged visual
positions. Reproducing the native minimum-area, factor-32 geometry gives
480-by-2208 and 1,035 visual positions. Adding ten native chat positions
explains its observed `prompt_eval_count=1045`. The same calculation explains
all 24 image-only counts. Across the sample, the custom reader has 20–352
visual positions, while the reconstructed native budget is 1,024–1,152.

This post-hoc [geometry audit](evidence/native_visual_geometry_v70_result.json)
binds the actual runtime log, raw response counts and inspected local
llama.cpp source revision. It does not verify the actual resized RGB pixels,
the numerical visual features or a causal accuracy effect of resizing.
The inspected source tree and executed native binary have separately recorded
provenance. The native-source comparison is therefore not matched to the
ILM's internal visual sampling budget.

Upsampling creates no new textual information. It changes how existing strokes
are presented to the pretrained patch encoder. A controlled image-interface
test must distinguish this scale difference from chat/task structure, imported
weight compatibility and output-head learning before another long adaptation.

## Execution, verification and scope

The 96 requests complete in 94.037 seconds including source startup and cleanup.
Maximum sampled GPU 0 resident memory is 7,566 MiB; this is NVML residency, not
the PyTorch allocation metric used for the custom actor. The owned server and
its runner exit, port 11437 closes, and GPU 0 returns to 16 MiB. The shared
LocalLLM service on port 11434 and GPU 1 work are unchanged.

Independent verification takes 0.482 seconds on CPU. It reloads original
source text, verifies all image bits and all request bodies, parses raw replies,
recomputes edit scores with an independent vectorized recurrence, checks all
eight paired outcomes and verifies resource cleanup. The fixed source archive
SHA-256 remains
`1329cd5ab37e5ef6faef37cd8ff8660ffc96ed5b47b34fddf8eb0912e6c82466`.

The [protocol](native_visual_baseline_v70_protocol.md),
[measurement receipt](evidence/native_visual_baseline_v70_measurements.json)
and [exact verification receipt](evidence/native_visual_baseline_v70_result.json)
record the evidence. Raw requests and replies remain under
`artifacts/native_visual_baseline_v70_20261105/attempt01/`. Query report SHA-256:
`a892c8553e819fb481b61f38a468d257c3a6d4281ed1f4f3648f5a83fa394f34`;
score report:
`6f4a3230d06262c07091abd83a44f55665a97a13d83ea378cce0442a70d4f543`;
verification:
`71926b423d97dab6f5ee99ff7b94633ad2919f18a4fbfddd806b27a057a0a35e`.

Reproduce with `scripts/native_visual_baseline_v70.py` and a fresh `--out`
directory: run `--stage prepare` with CUDA hidden, `--stage query` with
`CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID`, and `--stage score`
with CUDA hidden again. Then run
`scripts/verify_native_visual_baseline_v70.py --out RUN`. The geometry audit
uses `scripts/audit_native_visual_geometry_v70.py --run RUN --out NEW_JSON`.

The immediate research requirement is to preserve the source's visual-language
capability through the independent image path. This result neither proves
that the custom reader has lost all semantics nor selects a unique repair.
Useful image-to-image language remains unachieved, and the original V68
rejection is unchanged.
