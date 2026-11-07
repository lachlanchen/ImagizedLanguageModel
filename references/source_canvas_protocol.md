# Can source visual features rescue unseen canvas writing?

Prospective comparison, 8 November 2026. The preceding whole-canvas writer
fits 26/32 training answers but fails all twelve new-copy requests and all
128 normal development answers. Its complete prospective fit rule fails.
This comparison changes the information supplied to the writer, rather than
extending that failed 32-example recipe. No general ILM capability is assumed.

Both arms start from identical new weights and use the same 64-position,
six-layer, 384-wide continuous canvas decoder and fixed image codec. The
summary arm attends to the frozen reader's 52 final task-frame states. The
source arm additionally attends to the existing per-ink-unit visual features
and their four real geometric coordinates. Each source feature averages nine
samples from the frozen final visual field. These are **pooled descriptors**,
not all visual positions, source pixels, native KV history, character IDs or
codec coordinates. Pooling may discard the detail needed to spell a glyph.

For summary S, source features V, geometry G and white-image code w:

    H_summary = P(LN(S)) + t_summary
    H_source  = concat(H_summary, P(LN(V)) + A(G) + t_source)
    Z_arm = LN_768(w + W T(Q, H_arm))
    Y_arm = threshold(D(Z_arm))

P, A, the two type vectors, Q, T and W have identical initial tensors in the
two arms. Only the input feature route differs. The summary computation does
not use V or G; unused source-only parameters remain in its optimizer for
transparent capacity accounting. Both produce the entire 32-by-2048 canvas
without target lengths, previous answers, a glyph inventory, source crops,
OCR, a language output head or a teacher service. Conditional MSE remains a
deterministic prediction objective and can average incompatible answers.
This comparison does not resolve that separate distributional limitation.

Use all 3,257 previously admitted image pairs: 3,193 CMRC2018 Chinese short
extractive answers and 64 prepared Chinese/English copy requests. Preserve
the earlier eighteen exclusions and their original reasons; add no new
selection. Authenticate original images, reference PNGs, cached feature
digests and input bindings. Each training case has one accepted reference.
Encode complete reference pixels with the fixed codec, padding exterior white
to 32-pixel tile boundaries. A target exceeding canvas capacity is a hard
preparation error. Keep a supplied-code reconstruction score for every pair;
it is an answer-assisted control, not autonomous writing or a capacity bound.

Each arm receives exactly 3,257 AdamW updates, batch four: four complete
shuffled passes, 13,028 draws. Seed 2026110802, exact sampler seed +1. Peak
learning rate 3e-4, 100-update warmup, cosine decay to 3e-5, betas (.9,.95),
weight decay .01, gradient norm clip one, no dropout. The loss averages MSE
over active tiles and exterior padding separately with equal weights. No
development-based checkpoint selection or additional training after results.
Before production, check actual finite gradients in each used group and exact
optimizer/sampler/model resume. Discard preflight updates before both arms.
Serialize current and immediately previous optimizer, sampler and RNG states.

After each arm finishes, generate all 224 previously prepared development
cases: 96 normal Chinese questions, their 96 question-erased counterparts,
and 32 held-out Chinese/English copy requests. Also retain the first 32
training outputs in the fixed inventory order. The development cohort differs
from the preceding 128-question canvas screen; do not compare their totals as
the same cohort. This public development has been used repeatedly, not sealed.
Generate without opening references, then score every accepted reference at
the common left origin with exterior-white padding, retaining all raw pixels.
Report exact images, foreground F1, both-correct question pairs and the
normal-minus-erased gain separately. Inspect fixed complete output sheets.

A source-feature lead alone is not useful language performance. A continuation
screen requires at least 24/32 exact unseen copies, 32/96 exact normal QA,
17/48 both-correct QA pairs, at least .10 normal-minus-erased F1 gain, and at
least .10 F1 improvement over summary-only in both normal QA and copying.
These are practical decision floors, not statistical significance tests or
full-ILM qualification. If neither arm clears the screen, do not extend this
unchanged pooled-feature/MSE recipe merely because training loss decreases.
Any next treatment needs a separately stated representation or objective
hypothesis. Short extractive QA cannot establish free language composition.

This is initially a **cached-feature diagnostic**, not an independently
qualified native-image product. It reuses 11 GiB of source feature files and
their inherited Qwen3-VL pretraining. Report neural training time separately
from original feature preparation and pretraining. A positive result would
still require actual PNG-input replay and a clean inference bundle before
claiming an executable independent image path. The complete inherited reader
has billions of parameters; the new head alone is not the full lightweight ILM.

Use GPU0 only under the existing project lock, sequential arms, one persistent
owned controller, no GUI. Preserve interrupted evidence and exact resume
state. Commit source before execution. Bind every final checkpoint, code
coordinate tensor and PNG by hash; verify scoring independently on CPU.
Keep the overall ILM goal active and unachieved.

CMRC2018: Cui et al. (2019), CC BY-SA 4.0,
<https://github.com/ymcui/cmrc2018>. Qwen3-VL: Alibaba, Apache-2.0.
