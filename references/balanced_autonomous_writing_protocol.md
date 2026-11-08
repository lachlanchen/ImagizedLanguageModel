# Balanced exposure after the complete additional-training control

The final-image pilot exceeds the newly completed old-objective control by
three held-out OCR-exact copies, meeting their exploratory comparison rule.
All net held-out gains are English; Chinese remains 3/16 in both. The pilot
uses only 13 Chinese draws covering nine of sixteen available Chinese training
inputs. Test a bounded continuation with complete balanced example exposure.
This is additional learning, not a matched causal test of the curriculum.

## Unchanged model and loss

Start from the completed final-image pilot checkpoint, SHA256
`75e192a265069bdebe0b1e07de5339ce814631ccdbef8751b3e1ea75c16f5e5d`.
Retain its rank-16 adapters, full externally pretrained undistilled generator,
frozen native VAE, fixed generic condition, native-autocast input encoding,
full 50-step sampling, guidance one and 512x256 output geometry. Use the same
first 32 training copies, sixteen Chinese and sixteen English. Do not use
extra training strings, natural-QA labels, held-out examples or font changes.

The loss remains exactly the verified `final_image_loss`:

    q = G_theta(input_image, independent_noise)
    L = CE(frozen_reader(q), target_string+EOS)
        + 10 * balanced_ink_background_L1(unclipped_decoded_RGB, target_RGB).

No answer is supplied to generation; it enters only this final-image loss.
All 50 sampler steps remain in the gradient path. Preserve the canonical
style limitation. No OCR repair, font lookup or varying text condition enters
native deployment. These existing algorithms are not a new physics theorem.

## Fixed schedule and resumable execution

Run 128 additional updates: four complete permutations of the 32 original
training IDs. CPU seed 20268502 generates each permutation. Independent CPU
seed 20268503 generates one noise seed in [0,2^31-1) per update. Each of the
32 examples therefore receives exactly four updates, for 64 Chinese and 64
English draws. Model seed is 20268501. This changes exposure as well as
training duration; isolate neither as a causal effect.

Restore the pilot's FP32 AdamW state as well as its adapters. Retain learning
rate 1e-5, decay .01 and global clipping one. Optimizer step counts continue
from 32 to 160. Commit this implementation and protocol before execution.
Persist the complete deterministic schedule, each actual pre-update image,
loss, IDs, seeds and an atomic checkpoint. Resume only that unchanged schedule,
source, optimizer and completed step. Do not choose among checkpoints using
held-out quality. Preserve interrupted artifacts rather than rerolling seeds.

Before optimization, reproduce the first two saved pilot outputs bit-exactly
through the native inference helper. On the first actual differentiable
training call, also compare final pixels to native inference at its scheduled
seed before loading the target. Check finite nonzero prediction gradients at
steps 0,24,49, finite nonzero adapter gradients, no frozen parameter gradients,
and changed adapters. Use GPU0 alone with the shared lock; fail on nonfinite
values or 22 GiB allocated memory. There must be no other project-owned heavy
job while it runs. The prior observed update cost implies about 75 minutes;
this estimate is not a measured result for the new run.

## Full assessment and prospective decision

After all 128 updates, generate all 64 existing input-only evaluation records
in original order, with their original seeds, final weights and native
settings. Use a fresh process and unchanged CPU evaluator/reference calibration.
Authenticate every PNG against actual neural PT pixels and review every case,
including all Chinese failures. Report training and separate assessment costs.

Exploratory continuation support requires at least 6/16 held-out Chinese
OCR-exact copies (three more than the starting pilot), no decrease below
15/32 seen copies overall, and no decrease below 10/16 held-out English.
Report the counts regardless of this rule. A rule failure does not earn
unchanged scale-up. Writing qualification still requires 30/32 correct,
legible requested strings alone in each split and raw review. OCR alone is
insufficient. This reused development cohort is not a new blind test.

No semantic-QA claim is admitted. Subsequent language assessment must change
the question about the same unseen passage and require the appropriate
answer, with content controls. Independent general language, historical forms,
lightweight foundation training and matched-quality efficiency remain open.
