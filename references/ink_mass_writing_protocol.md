# Controlled physical-loss replacement after failed balanced writing

The complete balanced continuation fails its Chinese and seen-retention
requirements. Do not repeat unchanged scale-up. A CPU audit shows that
reference-derived duplicate lines cost a median 1.790% of deletion under the
current physical loss. Test one explicit correction rather than adding model
capacity, new data, or a new reader. This is a copying repair experiment; it
cannot establish language understanding.

## Fixed intervention

Both arms start from the complete balanced checkpoint
`fc031796540a4b6ca4c86774a5033e94ae32b415e825b6a221128756748d3575`,
restoring its adapters and AdamW states at step 160. Keep native 50-step
sampling, independent noise, BF16 input encoding, guidance one, fixed generic
condition, 512×256 output, frozen base/VAE/reader, rank 16, learning rate
1e-5, decay .01, clipping one and the same first 32 canonical training copies.
No new font, target, natural QA, retrieved answer or inference-time OCR enters.

Control uses unchanged `final_image_loss`: reader CE plus ten times balanced
unclipped RGB L1. Treatment replaces only this physical term with
`coefficient * ink_mass_kl`, defined in `ilm/image_edit/ink_mass.py` and derived
in `writing_physical_loss_audit.md`. Temperature is .05. Its fixed target
normalizer and same affine-logistic mapping for prediction and target retain
the exact-reference optimum, with a gradient on unclipped saturated errors.
It is standard Bernoulli KL, not new physics or a language theorem.

Set the coefficient **before any optimization** to the median over only the
32 training references of `10 * B(blank,y) / KL(blank,y)` in the committed
CPU audit. This gives **0.4035679875746343**. It matches the scale of the
reference-deletion cost rather than using held-out performance. Do not tune
temperature, coefficient or budget after observing output quality. This
calibration does not match parameter-gradient norms or training compute.

## Real bounded learning and checks

Run the two arms sequentially, control first, with the same 32-update schedule:
one full permutation, sixteen Chinese and sixteen English. Model seed is
20268601; CPU order seed is 20268602; noise-seed stream is 20268603. Restore
the same step-160 optimizer for each arm; all states must end at step 192.
No optimizer reset, inherited control weights, checkpoint selection or rerolls.

Commit the runner, module and protocol before launch. Reproduce the first two
retained balanced-checkpoint outputs bit-exactly, and compare the first actual
differentiable forward with native inference before loading its target. Require
finite nonzero trajectory gradients at steps 0,24,49, finite adapter gradients,
no frozen gradients and changed adapter tensors. Record pre-clipping norms,
each actual pre-update image, full source hashes and an atomic checkpoint.
One failed arm invalidates a claimed completed comparison. Preserve failures
and partial results; no automatic restarts. No separate discarded optimizer
preflight is needed because these checks run in the actual first update.

Use GPU0 alone and the shared project lock, sequentially with model/publication
jobs. Preserve GPU1 and unrelated projects. Stop on nonfinite loss/gradients
or at 22 GiB allocated memory. No assertion that fitting memory is meaningful
language performance.

## Complete assessment and decision

Generate all 64 original input-only records from each final checkpoint with
the original seeds and native settings, through separate processes. Use the
unchanged independent CPU OCR evaluator and clean-reference calibration. Bind
every PNG to actual neural tensors, retain every case, inspect the raw pages,
and report all seen/held-out and Chinese/English counts with costs.

Exploratory repair support requires **all**: treatment held-out Chinese at
least 6/16 and at least two more than control; seen overall at least 13/32 and
no worse than control; held-out English at least 13/16 and no worse than
control. Report paired gains/losses and visible extra writing, omissions and
duplications. This rule does not replace the unchanged 30/32 correct, legible
requested-string-only writing floor in each split. OCR is not manual accuracy;
the cohort is reused development and no independent-seed replication exists.

Even a favourable result is only support for a physical-writing repair.
Language learning still requires unseen passages with different questions and
appropriate independent answers. The already audited CMRC pairs are available
for that separate evaluation; this experiment does not train or score them.
