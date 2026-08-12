# Spatial Retinal Motor Plan V19 Protocol

Date preregistered: 2026-08-12

Status: fixed before V19 model implementation or training

## Question

V18 proves that a global `192`-dimensional image-derived intent can drive a
small deterministic planner to produce recognizable simple and medium Chinese
writing. Dense forms still merge or lose components. V19 tests the narrow
causal hypothesis that the missing information is **spatial retinal topology**,
not a need for tokens or a much larger image generator.

## Student Boundary

The V19 isolated writer may receive:

- the frozen retina's normalized global continuous state from a different-font
  semantic image;
- the frozen retina's continuous `4x4` internal feature field from that same
  semantic image; and
- a continuous style image containing a different character.

It may not receive target ink pixels as a condition, strings, token IDs,
Unicode IDs, OCR, character labels, a finite glyph/visual codebook, lookup,
candidate classifier, or external language model. Target ink is loss-only.
Offline evaluators may use images, source metadata, and answer keys, but none of
them may enter training or inference.

The spatial field is not the final language solution. In this isolated actuator
test it is observed from a different-font image. A later causal visual field
must predict it from image history before autonomous generation is claimed.

## Architecture Intervention

Let the frozen retina expose

\[
z=R_{\mathrm{global}}(x^{\mathrm{semantic}})\in\mathbb S^{191},
\qquad
F=R_{\mathrm{field}}(x^{\mathrm{semantic}})\in
\mathbb R^{192\times4\times4}.
\]

V19 adds a zero-initialized continuous spatial residual to V18's learned global
seed:

\[
H_0=H_{\mathrm{V18}}(z,E_s(x^{\mathrm{style}}))
+\sigma(g)A_\phi(F).
\]

All compatible V18 parameters are warm-started exactly. `A_phi` is initialized
to zero, so the initial V19 function equals the selected V18 function. Any
improvement can therefore be attributed to learned use of the new field more
cleanly than in a from-scratch comparison.

## Image-Only Complexity Strata

Complexity is computed only by image geometry on the binarized target image
`b = 1[x >= 0.5]`. It uses no character identity or text label:

\[
C(x)=\operatorname{mean}(b)
+0.5\left[\operatorname{mean}|\Delta_x b|
+\operatorname{mean}|\Delta_y b|\right]
+0.1\operatorname{mean}\left[
\operatorname{AvgPool}_{4\times4}(b)>0.05\right].
\]

The fixed strata are:

- simple: `C < 0.24`;
- medium: `0.24 <= C < 0.35`;
- dense: `C >= 0.35`.

These thresholds were calibrated once on training-side render geometry before
V19 implementation. Representative mean scores were `王=0.228`, `子=0.195`,
`而=0.307`, `作=0.255`, `將=0.348`, `開=0.375`, `緯=0.381`, `劉=0.398`, and
`鬱=0.429`. These characters do not enter the student or define evaluation
membership; only the equation does.

## Causal Branches

Every development validation must render these branches with the same style:

1. correct global state plus correct spatial field;
2. correct global state plus batch-shuffled spatial field;
3. batch-shuffled global state plus correct spatial field;
4. both global state and spatial field batch-shuffled; and
5. correct global state plus a zero spatial field, the matched V18-like
   ablation.

The report must include overall and complexity-stratified pixel F1 for all five
branches, retinal retrieval, target cosine, ink occupancy, condition pixel L1,
and spatial-residual magnitude. A gain is causal only if the correct branch
beats the spatial-shuffled and zero-field branches while target and style stay
fixed.

## Development Selection

Partition salt: `spatial-retinal-motor-plan-v19`.

The primary score is dense-stratum correct pixel F1. A checkpoint is eligible
only if all of these prospectively fixed gates pass:

1. overall correct pixel F1 `> 0.68`;
2. dense correct pixel F1 `> 0.58`;
3. dense correct-minus-spatial-shuffled F1 `> 0.12`;
4. dense correct-minus-zero-field F1 `> 0.03`;
5. overall correct identity top-1 `> 0.75` and above both-shuffled top-1;
6. correct target cosine `> 0.84` and above both-shuffled cosine;
7. correct-versus-both-shuffled condition pixel L1 `> 0.08`;
8. semantic-reference versus target pixel L1 `> 0.05`; and
9. no frozen image is instantiated.

Ties are resolved by overall correct pixel F1, then the earlier step. Selection
uses development images only.

## Blinded Readability Gate

Before any frozen query, create one deterministic 48-example development audit
with `16` simple, `16` medium, and `16` dense outputs. For each output, show
eight randomly ordered different-font candidate images without text labels or a
visible target row. Exactly one candidate group is correct; visually duplicate
forms are accepted using the frozen retina's precomputed duplicate mask.

The reviewer records one candidate number or `unreadable` without seeing the
answer key. Required recognition is:

- simple: at least `14/16`;
- medium: at least `12/16`;
- dense: at least `10/16`;
- total: at least `38/48`.

The answer key and decisions must be stored separately, then scored by code.
Changing thresholds, replacing examples, or opening a frozen bank after a
failed audit is forbidden.

## Frozen Policy

V18's unopened frozen bank remains sealed and is never reused for V19. V19 uses
a new salted partition. Frozen evaluation is forbidden until automatic
development gates and the blinded readability gate both pass. If either fails,
the result is documented as development-only and V20 receives a new protocol.

## Claims Allowed After Development Success

Allowed:

- a continuous spatial retinal field causally improves dense writing topology;
- the isolated writer remains compact and token-free; and
- measured compute and quality on the named development protocol.

Forbidden:

- autonomous language generation;
- general language understanding;
- Qwen or LLM parity;
- efficiency superiority without matched end-to-end quality;
- historical-form factuality from synthesized pixels; and
- a frozen result before the fixed gates authorize it.
