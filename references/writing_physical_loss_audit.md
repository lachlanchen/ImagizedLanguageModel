# Extra ink is weakly penalized by the current physical loss

A complete CPU-only audit of all 64 canonical reference pages finds a concrete
weakness in the final-image objective. Duplicating an entire correct line costs
only **1.790%** of deleting the answer at the median under its balanced L1 term
(range **0.837–3.447%**). These are explicitly reference-derived corruptions,
not generated outputs or a new model evaluation.

For a black/white error at one pixel, the loss weights are

\[
w_{\mathrm{missing}}=\frac1{2N_{\mathrm{ink}}},\qquad
w_{\mathrm{extra}}=\frac1{2N_{\mathrm{background}}}.
\]

The large blank page makes extra ink inexpensive per pixel. Multiplying the
whole term by ten preserves this imbalance. A training reader can also ignore
duplicated writing, as documented in the [earlier reader probe](autonomous_writing_loss_result.md).
Together these observations motivate a correction; they do not prove this is
the unique cause of actual training errors. The original loss still has an
exact-reference minimum and does not mathematically require duplicate lines.

## Candidate correction and its limits

Define a smooth ink probability separately for every RGB sample:

\[
a(p)=(.5-p)/\tau,\quad t=\operatorname{sigmoid}(a(y)),\quad \tau=.05,
\]
\[
L_{\mathrm{ink}}(p,y)=
\frac{\sum_j \{\operatorname{BCEWithLogits}(a(p_j),t_j)
-\operatorname{BCEWithLogits}(a(y_j),t_j)\}}{\sum_j t_j}.
\]

This is summed Bernoulli KL divided by **fixed target** ink mass. The same
invertible map is applied to predicted and target RGB, so the exact target
remains the unique optimum in real arithmetic; anti-aliasing is not assigned
a different optimum. No predicted ink mass or clipped training pixels enter
the denominator. Its derivative is

\[
\frac{\partial L_{\mathrm{ink}}}{\partial p_j}
=\frac{t_j-\operatorname{sigmoid}(a(p_j))}{\tau\sum_k t_k}.
\]

On the same 64 diagnostic pages, duplicate-to-deletion cost is **85.857%** at
the median (range **82.079–91.566%**). Finite differences agree with autograd;
the exact reference has zero loss and zero gradient. Wrong saturated white
and black fields receive correctly directed nonzero gradients. These checks
validate the calculation, not its optimization behavior in the writer.
This uses the standard stable [BCE-with-logits formulation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BCEWithLogitsLoss.html);
it is not a new physics result or a language-learning theorem.

The audit uses clipped, quantized PNGs. Historical training uses unclipped
RGB before display normalization, and also a reader loss. Therefore these
numbers do not reconstruct the full historical training objective. Loss
magnitudes across the two functions are not directly comparable; any training
comparison must fix its coefficient and gradient checks before optimization.
Reducing duplicated lines alone would not demonstrate semantic understanding.

Reproduce with:

```bash
../LocalLLM/.local/image-generation/venv/bin/python \
  scripts/audit_writing_physical_loss.py \
  --output references/evidence/writing_physical_loss_audit.json
```

The [complete report](evidence/writing_physical_loss_audit.json) includes all
64 cases, source-image hashes, both diagnostic corruptions' scores, actual
starting/final PNG scores and gradient checks. It loads no neural model,
makes no optimizer updates and does not repair any generated page.
