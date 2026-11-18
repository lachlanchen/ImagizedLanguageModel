# A pictured writer can change who speaks and whether permission is granted

On a real held-out sentence from *Kongzi Jiayu*, the image writer read 子曰： 不可。 and drew 孔子说： 不可以。 The source and the entire answer were absent from the corresponding sides of its training corpus. Changing the speaker to 王 and removing the negative gave 王说： 可以。 Restoring the negative gave 王说： 不可以。 The output is a sequence of 32 × 32 bitmaps. No teacher, text decoder or Unicode lookup is needed for image inference. [Figure 62](../publication/ilm-nature/figures/Figure_62_visible_likelihood.png) · [Standalone inference](../models/visual_edit_zh/README.md)

## What transferred

We changed four speakers and two polarities in this one speech construction. All eight source images and all eight complete neural answers were absent from the training data on their respective sides. The neural writer preserved speaker and permission or refusal in all eight. Seven whole modern answers were acceptable. For 子曰： 可。 it supplied an extra “we”. The phrase composer produced six acceptable answers; both methods got both polarities right as complete answers for three speakers. The construction was selected after finding the original exact answer. A study-involved model reader performed the masked assessments. This modest result supports generalization across visually expressed speaker and polarity substitutions within that construction. It leaves general Chinese comprehension open. [Original images, masked reading, assignments and training-overlap checks](evidence/visible_likelihood_20261119)

## What failed on broader language

In 1,024 further natural passages from 423 unseen chapters, the all-path writer made 17,940 glyph edits over 25,610 reference glyphs. Its matched single-path control made 18,276, and the identical parent made 20,461. The paired improvement over single-path training was 1.31 percentage points (95% chapter-bootstrap interval 0.03–2.56). Copying still did better at 16,660 edits, and the phrase composer did better at 15,787. Both new models achieved nine exact glosses, four requiring a change. The same-length source exchange raised the all-path writer to 25,203 edits. On eighteen source-stratified passages the reader accepted zero neural answers, four composer answers and fifteen references.

In 秦使魏冉之赵，出兵助燕击齐。, Qin sends Wei Ran to Zhao. The neural writer made Wei Ran's army the destination. The phrase composer preserved the proper name, destination and military participants. An image-level objective can reward many of the right marks even when a wrong role changes the sentence. All four exact changed answers, all incorrect answers and the paired comparison remain in the saved evidence. [Quantitative results and case roster](evidence/visible_likelihood_20261119/scores.json) · [Masked judgments](evidence/visible_likelihood_20261119/grades.tsv)

## What was trained

Both arms began with the same recurrent visual writer. Every training decision predicts a visible image or END. For a known target bitmap the new objective sums probability from every matching source position and exemplar. Its matched control selects one alignment path. Both arms share the same alignment penalty, unknown-ink drawing loss, optimizer, examples and final pooled decoder. Training ran for four passes over 55,872 real classical–modern Chinese pairs on one RTX 4090 D. Each arm used 2,368 updates. The all-path arm took 50.4 seconds and peaked at 441 MiB of allocated tensors. The parent had already trained eight passes over the corpus in 93.4 seconds, reusing a previously fitted drawing component. These are incremental costs, and no matched-quality efficiency ratio against an LLM has been demonstrated. [Protocol](visible_image_likelihood_protocol.md) · [Fit and checkpoints](evidence/visible_likelihood_20261119/fit.json)

An initial label table for the eight targeted prompts paired classical images with the wrong textual field. We discarded those ungraded labels, kept all original generated pictures, and reconstructed the packet from the rendering font. Every corrected cell matches the original bitmap exactly. Every displayed character in the separate eighteen-passage reading also passed an independent render-back comparison. The raw paired training and broad evaluation streams live under `artifacts/aligned_visual_translation_v1/` and `artifacts/visible_image_likelihood_v1/`. [Corrected speech packet](evidence/visible_likelihood_20261119/speech_review.json)

## Why the objective changes

Let a latent route have score equal to its negative log probability, conditional on the source and the previously written ink. The model can produce one visible glyph along several paths. Its effective energy is the negative log of the sum of the paths' Boltzmann weights. This is the same log-sum-exp operation used to aggregate microscopic states that share an observation. Differentiating it credits each matching path in proportion to its posterior probability. For a designated path among the matching paths, its excess loss over the pooled loss is the negative log of its posterior share. The comparison against the single-path arm tests this consequence directly. It gives a small measurable benefit to glyph accuracy; the role of envoy and destination still requires evidence at the level of whole sentences.

## Reproduction

With the aligned corpus, earlier held-out rosters and shared parent available, the following entry points reconstruct the matched training and fresh-chapter evaluation in a fresh experiment directory. Each evidence-producing command preserves existing runs. The independent inference example in the model README uses only published code and weights.

```bash
PYTHONPATH=. python scripts/train_visible_image_likelihood.py
PYTHONPATH=. python scripts/evaluate_visible_image_likelihood.py
PYTHONPATH=. python scripts/evaluate_visible_speech.py run
```

The four Figure 62 CLI reproductions used CPU image input. Each packed answer and PNG exactly matched the corresponding evaluated pixels. [Replay receipts](evidence/visible_likelihood_cpu_replay_20261119.json)
