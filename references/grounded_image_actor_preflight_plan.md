# Grounded image actor: complete-model preflight plan

5 November 2026. V63 is complete and rejects its assignment intervention.
Its autonomous outputs remain largely incorrect and fragmented. The next
candidate reuses authenticated visual/contextual weights and learns a joint
continuous writer. This unit establishes executable model boundaries and
measured full-model feasibility; it cannot establish useful language.

## Scope

Implement `ilm/visual_lm/grounded_image_actor.py` and an actual-weight loader,
plus `scripts/check_grounded_image_actor.py`. Use the existing Qwen3-VL GGUF
reader and pinned source, without downloads or teacher-service calls. Preserve
all V63 sources, protocol, weights and raw outputs. Keep manuscript changes
outside this unit; a separate planned revision will report completed V63.

The actor receives physical prompt images and completed 32-by-64 writing
blocks. The prompt may be a losslessly folded training strip or an observed
page with dimensions divisible by 32. RGB normalization and patch order must
match the pinned native image processor. Each completed writing block is a
separate image before the bidirectional vision encoder. Continuous query
vectors request the next block; they do not index a language alphabet.

Use all 27 vision layers and all 36 contextual layers. Remove the contextual
vocabulary module. Inherited weights are frozen BF16; train rank-eight
adapters in contextual query/value/output projections, continuous queries,
an image flow writer and a structural stop head. Retain three DeepStack
feature sets and explicit spatial rotary coordinates. The writer generates
all pixels of a block jointly, with a conditional flow objective. A separate
binary stop event distinguishes termination from valid white space.

Teacher forcing inserts each block only after its prediction queries. A
causal contextual mask and independent block encoding must prevent future
answer pixels from changing earlier prediction states. Deployment generation
must reread only its own completed output pixels. No tokenizer, Unicode IDs,
language vocabulary, glyph bank, OCR or external model call is accepted in
the actor. Offline teacher alignment is a later training choice, not a hidden
component of this preflight.

## Actual checks

1. Revalidate source hashes and host/GPU ownership. Use physical GPU 0 only,
   one owned job, at least 24 GiB available host RAM and swap below 75%.
2. Match image normalization/patch order to the installed native processor on
   two fixed real training crops, including an asymmetric pixel pattern.
3. Materialize every retained parameter from the real archive in the complete
   actor. Record source/configuration hashes and actual parameter counts.
4. Execute a complete teacher-forced image-to-image loss, backward and optimizer
   updates. Check finite gradients in the writer, queries and first/last
   contextual adapters; frozen base weights must have no gradients.
5. Check future-block intervention and separate-image encoder isolation on
   real training pixels, plus cached versus full-prefix conditions. Declare
   numerical tolerances and report actual errors, without assuming exact BF16
   equality for different execution shapes.
6. Measure the largest complete training row's forward/backward at the chosen
   writer block budget and actual peak allocated/reserved memory. Do not
   substitute a truncated prompt or a smaller contextual body.
7. Save and restore trainable weights, optimizer and RNG state, comparing a
   continued update against replay with the same frozen base. Retain the
   generated image and loss traces as execution evidence, not language quality.

Preflight acceptance requires all applicable identity, boundary, finite-gradient,
causality and replay checks, and less than 22 GiB peak allocated memory. Exact
BF16 future-intervention equality is required for identical shapes; cached
versus full-prefix states use a registered relative maximum tolerance of 0.02
against the full-prefix maximum magnitude. If a check fails, preserve the
failure and repair the actual implementation before accepting feasibility.
No architecture or quality claim is inferred from a smoke loss decrease.

## Follow-on work

After actual feasibility, register a learning experiment with an explicit
image-writing curriculum, semantic transfer if used, fixed decoding and
matched controls. Run real training and autonomous Chinese/English answers.
Preserve inherited pretraining attribution and the full image-language goal.
The manuscript must continue to distinguish component execution, bounded
reading tests and general language capability.
