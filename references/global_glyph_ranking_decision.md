# A whole-glyph training comparison after the blank-mode diagnosis

The [completed probability probe](joint_context_blank_modes_result.md) rejects
the registered decoding-only eligibility rule. Blank output is a certified
global density mode at 108 false-blank positions; saved greedy choices reproduce
without disagreement. The next bounded intervention changes the training
objective. It does not ban empty output or add a font/character lookup, occupancy
override, forced ink pixel, beam search or Unicode classifier.

## Candidate objective

Let `q(g|h)` be the existing normalized 128-mask physical image density, `z` the
all-white glyph, and `a(g)` indicate any visible ink. Keep the complete-source
likelihood and balanced stop loss. Add one whole-glyph contrastive term:

```
L = (1/T) sum_t [ -log q(g_t|h_t)/128
    + a(g_t) softplus(1 + log q(z|h_t) - log q(g_t|h_t))/128 ]
    + 0.1 L_stop.
```

The ranking strength is one and the complete-glyph margin is one nat. Both
scores use the same actual target-preceding state. For a nonempty observed
target, this term lowers an empty image's preference relative to that target.
True whitespace receives its unchanged likelihood supervision and no anti-blank
ranking term. The density stays normalized, but the combined objective is not
pure maximum likelihood or a guarantee of calibrated probabilities. No language
advantage follows from the equation.

This is a measured-failure hypothesis: whole-glyph ranking may remove empty modes
that survived lower source likelihood loss. It can also overfit source forms,
damage legitimate whitespace or produce incorrect nonempty glyphs. Those are
outcomes to measure, not reasons to call fewer blanks a language success.

The idea of penalizing undesirable outputs during learning has precedents in
[unlikelihood training](https://arxiv.org/abs/1908.04319). That work evaluates
token/sequence degeneration and does not validate this physical-image loss.
[PIXAR](https://arxiv.org/abs/2401.03321) also studies a likelihood/generation
disconnection in pixel language models, using adversarial pretraining. Neither
paper establishes our proposed comparison, Chinese performance or one-GPU parity.
The expression above is a candidate contrastive ranking objective, not a new
physical theory or an application of either paper's exact algorithm.

## Implementation and necessary proof before a fit

The new [gradient function](../ilm/visual_lm/global_glyph_ranking.py) accumulates
source/blank head gradients in glyph microbatches and differentiates the actual
continuation graph once with the accumulated state gradient. It does not create
an optimizer or emit images. Independent CPU checks compare every context/writer
gradient against an unsplit calculation, including mixed and all-white targets,
zero ranking strength and non-divisible microbatches. These checks establish the
implementation on their fixtures, not language or full-body GPU behavior.

Register a new real-source preflight before neural execution: both original
complete five-glyph QA layouts and both maximal 513-glyph book layouts. Check
the new loss and all 131 actual learned-tensor gradients against unsplit QA
autograd, then take one actual complete-source optimizer step for each book
layout. Preserve frozen body, feedback, stop and affine buffers; reset every
update. Reuse the verified shared weights and process-local runtime. The old
preflight and its failed/finished attempts remain terminal and unchanged.

Only a successful fresh preflight permits registering a complete matched fit.
Initialize from the identical fresh head and point-final context used by the
completed context-gradient control. Keep the original 13,766 sources, one fixed
and two proportional presentations, 41,298 updates, 6,108,741 target exposures,
five fixed checkpoints, optimizer schedule, trainable tensors and greedy decoder.
Do not fine-tune a lucky intermediate or shrink the corpus to easy cases.

## Fixed evaluation and action

Before a fit, commit its exact implementation, source/weight/data bindings and
resource limits. Retain the unchanged 124-case physical cohort and original
90% requirement in every group. Future opaque review must derive only empty vs
nonempty flags from the original opaque pixels before its first freeze; it must
still actually inspect each board, preserve uncertainty and freeze before owner
unmasking. This prospective review safeguard cannot repair either old review.

Count legitimate source whitespace, false blanks, incorrect nonempty outputs,
canonical bitmap agreement, complete language and stopping separately. As an
additional prospective safeguard against replacing all blanks with ink, the
new candidate may not exceed the control's **11 false-ink outputs among 252
source-whitespace positions** in the complete supplied-history cohort. Keep all
seven groups visible; this is an additional control, not a revision of an old gate.
Fewer blanks or lower ranking loss alone do not qualify the model.

A physical/ink/whitespace pass permits the unchanged complete 708-stream
autonomous comparison with actual emitted-pixel feedback, paired questions,
erasures, long books and natural stopping. Preserve the existing 12/32 strict
proportional QA, 4/16 both-correct pairs, 24/32 readable QA and 4/12 coherent long
book pilot, earlier-success retention and dependence rules. Untouched
confirmation, independence, historical forms and whole-model matched-quality
cost are still required by the full goal.

If the new physical or whitespace condition fails, stop before autonomous
evaluation and use the retained outputs to redirect representation/learning;
do not extend unchanged epochs or retune a threshold. If physical conditions
pass but complete language fails, redirect conditional language learning rather
than claiming success from typography. No new fit has been executed by this
decision, and no useful independent visual language has yet been established.
