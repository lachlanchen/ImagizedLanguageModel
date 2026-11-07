# Matched contextual adaptation under an answer-image objective

Prospective comparison, 7 November 2026. The
[full backward preflight](contextual_pixel_learning_preflight.md) passes on
one 4090, including exact source-interface replay and optimizer resume. It
does not measure language improvement. This comparison tests whether learning
contextual attention improves held-out physical answer selection beyond
continued training of the same head on the same broader corpus.

## Fixed intervention and budget

Both arms start from the completed contextual pixel-span head
(`85369b1c1547f6fd02e5cc7ea601e202bb03ed8360e6e4964e94b5beec476a91`).
They retain the same full 7,531,277,171-parameter externally pretrained runtime,
numeric controls, frozen vision/base weights, source-pixel motor, image
geometry and 128-unit span law. Neither conditions on reference locations,
Unicode IDs, OCR or a teacher service. Training targets are physical PNGs.

Control updates only the 2,378,370 head parameters, using authenticated frozen
features. Adaptation updates that head plus 6,193,152 existing rank-eight
query/value/output attention parameters through the qualified deterministic
sampling adjoint. Cached control features must preserve the native BF16
interface. Before optimization, verify exact native-versus-cached logits,
loss and head gradients on the first admitted training case.

Use all 3,193 admitted QA images and 64 original training copies from the
completed preparation, with its 18 pre-call exclusions. No examples are
chosen from outcomes. For each arm use exactly four shuffled full passes:
13,028 draws, batch four by sequential gradient accumulation, 3,257 updates.
Use the identical fixed ID schedule, seed 2026110706. Every ID occurs four
times. Initialize both arms identically with fresh AdamW states; do not use
the preflight's changed tensors or optimizer moments.

Head peak LR is 1e-4; adapter peak LR is 1e-5. Use 100-update linear warm-up,
cosine decay to 10% of peak, betas (.9,.95), decay .01 and global norm clipping
at one over each arm's optimized parameters. Global clipping therefore
includes contextual gradients in the adapted arm; the tested intervention
is this complete adaptation regimen, not a unique internal causal mechanism.
All learned parameters, contractions and accumulated gradients use FP32;
retain the original BF16 descriptor/task-memory boundary. Use the final
checkpoint only. No development calls, outcome-dependent early stopping,
hyperparameter tuning or intermediate checkpoint selection during training.

## Data, execution and retention

Bind the completed preparation and preflight report hashes, all source files,
initial trainable tensors and exact schedule to the new run identity. Keep
completed prior experiment files immutable. Build source features without
opening references; reuse an earlier feature only when its original input
hash agrees and the payload authenticates. Targets remain a separate loss
input. Exact source spans may be found efficiently by matching each target's
physical width against source-unit boundaries; check equality with the
original exhaustive image-marginalization matcher before optimization.

Run the two arms and assessment sequentially under the project GPU0 lock.
Never use GPU1. Reuse the shared pretrained archive; save only learned
tensors, Adam moments, schedule cursor and CPU/CUDA RNG. Checkpoint every
32 updates and retain current/previous. Resume only an explicitly selected
existing run with unchanged source/data identities. Preserve journals of
aborted invocations and count their additional compute rather than presenting
replayed work as free. An observation timeout does not justify a new job.

The measured single-example backward cost is about 2.4 seconds. The adapted
arm is therefore expected to take about nine hours, plus feature preparation
and assessment; this estimate is not a measured final runtime. Report actual
cache/control/adaptation/native-assessment costs, peak allocated/reserved
memory, full runtime and optimized parameter counts. Head-cache costs and
external pretraining must not be hidden in efficiency claims.

## Fixed autonomous assessment and decisions

After both training arms finish, run all original 224 development inputs
and the same first 32 earlier training QA images through the complete native
image path for each arm (512 total calls). Open no references during these
calls. Keep every unmodified PNG and chosen source boundaries. Score original
pixel equality after removing only all-white horizontal exterior, with
unordered column F1 as a separate nonsemantic diagnostic. Reconstruct every
output independently from its declared source columns. Report both-correct
paragraph pairs and all question/passage erasure outputs.

The original four requirements remain: at least 30/32 new copies, 48/64
normal QA answers, 20/32 both-correct pairs, and at least 25 percentage points
normal-minus-question-erased exact accuracy. Passing them qualifies only
this prepared-strip extractive component. Separately call the adaptation a
useful improvement over control only if it adds at least 8/64 exact normal
answers and 4/32 both-correct pairs, while losing no more than two correct
copies. Report the complete paired outcomes even when either rule fails.
These floors are decision rules, not statistical significance claims.

This cannot generate an answer absent from its input. It does not validate
free image-language synthesis, historical writing, handwriting, general
instruction following, a small model trained from scratch, Qwen parity or
efficiency at matched useful quality. The full independent ILM remains the
objective. Preserve negative outcomes and update the manuscript after the
completed comparison, rather than treating execution qualification as success.
