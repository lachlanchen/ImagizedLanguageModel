# Full-unit first-form function transfer

The two completed answer cohorts separate a source decision signal from the
physical image writer. This experiment tests transfer of that function; it is
not a new language-training claim. The unchanged balanced writer is the parent
(`dcac4ee2…`). A normalization audit changes no source decision on 128 already
observed first states, so this experiment keeps the architecture unchanged.

Use only the existing 3,176 training-question first states. Every fifth question
is withheld as a synthetic calibration base; it was used by the inherited
upstream fit and is not a held-out language example. No observed development or
reserved answer state enters this experiment. The existing offline mapping
admits 126,213 source units with a supported first complete form, including
multi-character units. Preserve all excluded/terminal/partial counts. This is
broader than the previous 4,087 single-form anchor experiment.

For each admitted source unit, draw two training bases and one withheld base.
Apply at most 24 diagonal-variance-weighted, highest-competitor half-space
corrections with margin 1. The temporary source matrix is BF16-rounded, then
multiplied in FP32 without TF32. Recheck the complete source argmax and retain
only correctly verified projections. Record every failed unit and accepted
coverage; do not silently assume convergence. Projections are counterfactual
vectors, not demonstrated natural-image states or additional knowledge.

Start from the completed balanced checkpoint. Train only continuous G and D's
input MLP; retain E, convolutional D, stop, normalization buffers and all body
parameters exactly. Use pixel reconstruction alone, without the semantic-anchor
MSE that accompanied the earlier failed compiler. The balanced pixel loss is
the existing equal foreground/background BCE. It remains a marginal loss and
does not guarantee coherent or correct output; earlier joint-pixel experiments
also failed and are not erased by this choice.

Fix 18,000 AdamW updates, learning rate 1e-4 with cosine decay to 10%, weight
decay .01, gradient norm cap 1, seed 202611122 and BF16 training autocast. Each
256-state batch contains 128 synthetic, 64 gold-labelled training-first, 32
gold-labelled later and 32 genuine-copy training states. Sample synthetic forms
uniformly, then uniformly within their accepted source-unit/base states. Group
pixel-loss weights are .5/.25/.125/.125. Stop is retained without new targets.
Save periodic model/optimizer/RNG checkpoints; no best-held-out selection.

Qualification compares parent and candidate in actual FP32 runtime arithmetic
on every accepted withheld synthetic state, all 23,262 training states and all
256 existing cached held-wording copies. Preserve every binary image, F1 and
exact score and inspect fixed, unselected preview boards. The function gate
requires synthetic micro and form-macro F1 at least .95, at least 80% with F1
at least .95, training mean F1 at least .98, cached-copy mean at least .95 and
80% of copies at least .95. Cached copy states come from the unadapted source;
they do not replace a later actual adapted-reader copy assessment. This gate
does not assess natural stopping, fresh answers, language understanding or
unknown glyph identity. If it fails, archive the failure before another model
change. If it passes, register a third fresh-context cohort excluding both
previous cohorts and run unchanged-parent/actual-feedback answers separately.

The runtime architecture is unchanged: pixels → reader → continuous G/D →
actual pixels → E. Source projection, mapping, IDs, reference images and fonts
remain offline. The checkpoint contains neural weights, not a runtime lexical
classifier or glyph bank. The inherited 7.522-billion-parameter body remains
part of deployment and pretraining provenance; this is not training a small
foundation from scratch or a proven efficiency advantage over an LLM.

Use the existing GPU0 mutex and one stage at a time. Each stage has a 3,600s
wall cap, 1,800s cumulative pressure wait, 14GiB allocated CUDA cap, 48GiB host
entry and 32GiB live reserve, with existing .5/.1 memory-pressure limits. Do
not alter other projects, force swap eviction or overlap publication builds.
