# Whole-canvas writing from prompt-only visual memory

Prospective experiment, 8 November 2026. The previous goal turn completed the
verified manuscript update (`aad28c7`), which is progress in the evidence
record, not a model repair. No project model process is live before this unit.

## Hypothesis and computation

The V80–V82 writers corrupt later writing even under correct image history.
Their generated image embeddings replace a pretrained contextual body's native
continuation interface. This experiment removes that transition from the
writing computation: one learned decoder predicts the complete answer canvas
jointly from the same 52-state, prompt-only visual memory. It does not change
the inherited reader or the qualified continuous image codec.

For memory M(X), 64 learned continuous position queries Q, a six-layer
384-wide, six-head transformer decoder T, and fixed white-image code w,

    Z = LayerNorm_768(w + Linear(T(Q, Linear(LayerNorm_4096(M(X)))))).
    Y = threshold(D(Z))

The output consists of 64 adjacent 32-by-32 image tiles: one 32-by-2048
canvas. No reference length, preceding answer, language vocabulary, glyph
inventory, token decision, OCR, source crop or teacher enters generation.
All generated pixels are retained. The final projection initially vanishes,
so the initial code is the fixed white-image code. Model output is a
deterministic conditional estimate, not a normalized language likelihood.
An ambiguous conditional can still average visual alternatives; this
experiment does not assume that regression resolves that problem.

This is a composite architectural replacement, not an isolated causal test
of history alignment. In particular it removes causal image continuation,
adds repeated attention to memory, changes initialization and uses one-shot
code prediction. It is also not the earlier initial-state alignment learner.
The full runtime retains the externally pretrained 7.5B-class source body;
the much smaller trained decoder alone is not a lightweight independent ILM.

## Fixed fit and generalization screen

Use every one of the existing 32 training questions in sixteen distinct-answer
paragraph pairs, bound by `86809a7b29bc57c84ee4991d10d320e04031cf56da2ad6d49724e693aa20f884`.
Authenticate original physical prompts, cached prompt-only memories, complete
reference images and image-derived codes. Do not change or discard any case.
Pad unused canvas positions with the codec's white-image code. A target wider
than the fixed capacity is an explicit preparation failure, not a truncated
training answer. Code loss averages active tiles and padding tiles separately,
with equal half weights, then averages examples. No pixel/teacher loss is added.

Run exactly 2,000 AdamW updates, batch four, betas (0.9,0.95), weight decay
0.01, gradient clipping at one. Peak learning rate is 3e-4; warm up for 100
updates, then cosine decay to 3e-5. Seed 2026110801; shuffled exact-epoch
sampler seed +1. No dropout, development-based selection or best checkpoint.
First verify finite nonzero gradients after three actual updates and exact
optimizer/sampler/model resume; reset those updates before production.

Retain all 32 initial and final canvases, coordinates, references and source
bindings. The fit criterion requires at least 24/32 exact padded-reference
images, mean foreground F1 >= 0.98 and at least 12/16 both-correct pairs.
These are training-fit floors, not semantic or general-language qualification.
The fixed canvas has no learned stream-termination event, so its exact score
must not be presented as the earlier learned-stop exact metric.

Regardless of fit, assess all twelve pre-existing Chinese/English new-copy
requests and all 128 Chinese questions in the existing V68 development audit,
including their 128 matched question-erased images. Retain failures, alternate
accepted references, all 64 paired outcomes and references exceeding canvas
capacity. Do not train on those cases. Development has been reused previously
and is not sealed. Report copying separately from QA. A useful-reading screen
would require >=64/128 exact normal answers, >=23/64 complete pairs and a
>=0.10 normal-minus-erased foreground-F1 gain. No outcome alone qualifies the
full ILM. A failed training fit blocks scaling this unchanged canvas recipe.

The offline evaluator pads references with exterior white to the larger of
their width and the complete canvas width, preserving the common left origin
and every ink pixel. It reports exact pixel equality and foreground F1 over
that full space. References wider than the capacity remain in the denominator.
No display crop or cleaned transcription replaces a generated image.

## Execution and verification

Reuse existing authenticated prompt memories for the fit/new-copy screen.
New development memory must come from the actual unchanged image observer.
Keep one GPU0 job and its lock; reuse source weights and the matching NVIDIA
userspace wrapper. Source-body materialization is sequential with head-only
training. Never use or stop the other project's GPU1 runtime.

Checkpoint model, optimizer, sampler, CPU/CUDA RNG and source/data identities;
keep current and immediately previous resumable snapshots. Bind producer code
before execution. A clean inference bundle contains only the new decoder
configuration/state and provenance, plus the separately shared fixed codec
and source body. The image CLI must not open training data, feature caches or
references. Fresh PNG calls must reproduce at least the first two training
and first two new-copy canvases exactly from actual prompt pixels.

Independently verify source hashes, optimizer steps, data draw counts, all
retained PNG/coordinate pairs and pixel scores. Replay the frozen codec on
all retained coordinates. Report head-only preparation/training separately
from new source encoding, standalone setup/inference and external pretraining.
Preserve partial outputs on interruption; an execution check is not a language
result. Do not update publication outcome claims until this run is complete
and its raw writing has been inspected. The full goal stays active.
