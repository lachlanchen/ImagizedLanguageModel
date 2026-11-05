# Causal physical-image prefixes

Prepared 5 November 2026, before execution. This independent CPU preparation
leaves V63's active trainer, source pins, sampling and audit unchanged.

A pretrained vision encoder attends bidirectionally within its image. Feeding
a complete teacher-forced answer page through it can expose later writing to
earlier decoder states, even if the contextual decoder has a causal mask.
An image actor therefore needs visible prompts and completed output history
to reach the reader through an explicitly causal physical interface.

Implement and verify:

1. Fold the existing binary writing prompt strips into bounded-width pages
   without cutting through ink. Keep an exact inverse for the pixel audit;
   fail if a safe whitespace cut is unavailable. This host transform does
   not transcribe, tokenize or classify writing.
2. Prepare owning prompt/history tensors from a teacher-forced training row.
   Only observed prompt pixels and the specified completed blocks may be read
   or validated. Do not expose future answer length, content or source storage
   through the returned interface. Keep completed blocks separate so a frozen
   image encoder cannot attend across future output blocks.
3. Test exact pixel recovery, source-storage independence and invariance to
   changed/NaN future answer pixels on real cached training images. Audit all
   complete training prompts for invertibility and supported geometry.
4. With the authenticated cached Qwen3-VL vision encoder on CPU, demonstrate
   why encoding completed and future answer images together is unsafe:
   perturb only a future block and measure the preceding block's feature.
   Independently encoded completed blocks must replay exactly. Use fixed
   training images; do not inspect development or sealed cases.

For fixed observed X and completed images Y_<t, the intended visual state is
`E(X, Y_<t)`. It must be unchanged for every alteration of Y_>=t. This is an
information-flow requirement, not a new learning theorem. The test verifies
the physical interface and a specific encoder failure mode; it does not
establish language competence, train a new actor or select V63's successor.
