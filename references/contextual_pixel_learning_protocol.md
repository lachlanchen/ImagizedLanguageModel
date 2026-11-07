# Learning contextual image spans: execution preflight

Prospective protocol, 7 November 2026. The completed frozen-context experiment
copies 32/32 new strings but answers only 9/64 held-out Chinese questions.
Its 32/32 seen-question fit does not establish natural reading. The immediate
question is whether answer-image supervision can reach the native contextual
reader without changing the verified physical copying operation.

## Scope and hypothesis

Keep the authenticated Qwen3-VL visual weights, contextual base weights,
numeric task controls, image geometry, dynamic span distribution and exact
source-pixel motor. Initialize the head from the completed 3,200-update
checkpoint. Enable only the existing rank-eight query/value/output attention
adapters and the head. There is no tokenizer, character vocabulary, text
output projection, OCR, teacher service or answer location in conditioning.
The complete externally pretrained body remains part of the runtime.

This is standard low-rank adaptation applied to the existing image-span loss,
not a new learning theorem or a demonstrated repair. Prior work includes
[LoRA](https://arxiv.org/abs/2106.09685),
[PIXEL](https://arxiv.org/abs/2207.06991) and
[PIXAR](https://aclanthology.org/2024.findings-acl.874/).
An extractive success would still leave independent free image generation,
general instruction following, historical forms and matched-quality
efficiency unproved. The full ILM objective is unchanged.

## Exact forward and deterministic differentiation

The completed encoder uses nine bilinear samples per anonymous ink unit,
then caches visual descriptors and task states in BF16. Preserve those
forward operations exactly, including the BF16 interface. Native CUDA
`grid_sample` backward is nondeterministic. For a grid determined solely by
input pixels, sampling is a fixed linear map A from field positions to units.
Use the original `grid_sample` forward and the deterministic adjoint
`grad_field = grad_units.T @ A` in backward. A contains the mean of the nine
border-clamped, align-corners-false bilinear weights. No gradient to the
fixed physical grid is required. Validate the adjoint against CPU autograd
and finite differences in double precision, including border samples.
This does not alter the probabilistic objective or the forward representation.

## Preparation without model outcomes

Authenticate all 3,211 existing short CMRC training records. Render every
record with the same whole-line Pango renderer. Preserve all source annotation
alternatives that are representable as complete physical source spans;
exclude and enumerate rows with unsupported font height, cut ink runs or
unrepresentable references before any new model calls. An annotation is used
only offline to prepare target PNGs; the training loss finds all physically
matching spans from those PNGs. Include the existing 64 training copying
images without duplicating their files. Inputs and references have separate
indices. Verify all training article titles and context hashes are disjoint
from the unchanged development set. This remains reused public development,
not a sealed test.

Record every folded input geometry and ink-unit count. Determine transformed
sequence sizes from the actual image-budget function for each distinct shape.
The resource cases are the largest native sequence and the most ink units;
ties follow sorted source ID. Optimization checks use the first two admitted
QA IDs not used in the earlier 256-question training set. Selection never
uses model performance. Preparation may therefore reject more than the
earlier one excluded paragraph pair; count actual rows rather than assuming
the old exclusion count.

## Required checks before a longer comparison

1. Preserve completed source files and checkpoints. Commit the new module,
   runner and this protocol before execution; bind their hashes to receipts.
2. Four complete native calls (first two normal development questions and
   first Chinese/English development copies) must reproduce the completed
   cache's descriptors, task memory, units and geometry bit-for-bit before
   learning. They are execution calibration, not a new quality evaluation.
3. At zero adapters, enabling training/checkpointing must preserve the first
   calibration call's forward descriptors exactly. Use nonreentrant layer
   checkpointing; freeze the vision module and all base parameters.
4. Exercise real answer-image loss and AdamW on the two new QA images, with
   head LR 1e-4, adapter LR 1e-5, betas (.9,.95), decay .01 and clipping 1.
   Check finite gradients for every trained tensor; up matrices must receive
   nonzero gradients at initialization, and down matrices after an update.
   Every frozen parameter must retain no gradient and remain outside Adam.
5. With allocated Adam moments, exercise the largest native sequence and
   largest ink-unit case. Record full-runtime parameter counts, per-case
   times and peak allocated/reserved memory. Require peak allocated below
   23 GiB before authorizing a production run on GPU0.
6. Save only trainable tensors, optimizer and CPU/CUDA RNG. Continue one real
   update, restore, and require bit-exact trainable tensors, optimizer state,
   RNG and update metrics. These checks must finish before a long run.

Use seed 2026110705 and FP32 trainable parameters/contractions. Reuse the
existing shared source archive; never duplicate model weights. Hold the
project GPU0 lock and leave GPU1 and other projects untouched. Check host
RAM/swap, project processes, tmux and noVNC ports before loading. No GUI is
needed. Persist the process handle and terminal state; do not restart on an
observation timeout.

This unit implements preparation and execution qualification only. It does
not launch production optimization or choose a budget from development
scores. If checks pass, register a separate matched comparison with the
continued frozen-head control, identical new training examples/order and
final-checkpoint evaluation. Use the unchanged 64-question/32-copy assessment,
its question/passage erasures, both-correct pairs, raw images and existing
48/64, 20/32, 30/32 and 25-point requirements. No passing execution check is
a language-quality result. Do not edit the manuscript as if a repair worked.
