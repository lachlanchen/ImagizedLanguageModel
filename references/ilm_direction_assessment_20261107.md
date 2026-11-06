# Current direction: useful visual language generation remains unproved

Assessment date: 7 November 2026. The completed experiments do not establish
a useful ILM. The image-input/image-output research goal remains open; the
current learning recipe has not earned a scale-up. The unfinished V82 matched
comparison has no final quality result at this assessment date.

The latest completed [readout comparison](adaptive_readout_v81_result.md)
provides three distinct measurements on the same 32 seen training questions:

| Path | Adapted model result | What it tests |
| --- | --- | --- |
| Autonomous answer, generated-pixel feedback | 0/32 pixel-exact; 72.748% foreground F1 | Actual complete image generation |
| Correct preceding answer codes and known length | 0/32 pixel-exact; 79.629% foreground F1 | Conditional prediction with answer-history assistance |
| Supplied target codes through the decoder | 28/32 pixel-exact; 99.9953% foreground F1 | Physical representation and decoding of known answers |

Pixel equality is stricter than readability, and foreground F1 is not semantic
accuracy. The [complete raw galleries](evidence/adaptive_readout_v81_result/all_cases_24_31.png)
also show visible character corruption. This is not merely a strict score
rejecting otherwise legible answers. The fixed-case gallery includes short
outputs that look readable as well as longer outputs that deteriorate; neither
selected examples nor a mean pixel score establish useful language capability.

The decoder can represent the supplied targets accurately. The learned path
from a prompt and image history to the next image code remains inaccurate.
Correct history improves the score but does not repair writing, so accumulated
generated-history error is not the sole explanation. This localizes an observed
bottleneck; it does not identify a unique cause within contextual feature
learning, conditional prediction, the flow objective or its sampling procedure.

A [separate block-level audit](readout_pixel_margin_v81_result.md) finds that
the continued control's first-block F1 reaches 99.330%, while its later blocks
reach only 55.075% even with correct history. Of 21,400 incorrect later pixels,
10,665 have signed decoder margins below -8. Large wrong ink decisions remain;
small threshold or floating-point errors cannot explain all of them. These
are control-arm block averages, not the adapted complete-answer score above.

The current actor retains 7,583,225,971 complete parameters and trains
53,096,961. Its externally pretrained contextual body has learned a different
input/output interface. Attaching continuous image embeddings and an image
writer does not establish transfer of its original language-generation
competence. Fitting its adaptation within one GPU is an execution result,
not proof of a lightweight model, from-scratch training, efficient knowledge
learning or parity with a conventional 8B model.

The implemented flow objective predicts clean codes from a mixture of target
codes and noise under a supplied correct history. At inference, the target is
absent and sampling begins from noise. Accurate reconstruction of an assisted
near-clean input therefore cannot qualify free generation. The earlier
[time-coverage experiment](writing_times_v76_result.md) already demonstrates
that improved near-clean reconstruction can coexist with worse autonomous
answers. Merely changing the denoising-time schedule is not an established
repair.

There is independent evidence that language can be modeled in pixel space.
[PIXAR](https://aclanthology.org/2024.findings-acl.874/) generates rendered
writing and explicitly studies noisy output.
[MIXAR v2](https://arxiv.org/html/2604.11575v2) studies multilingual pixel
generation and trains a second stage on free-running outputs. Its reported
results do not establish modern 8B-model parity or single-4090 foundation
pretraining. This project already evaluated the released 477M stage-2 model
and attempted [full native adaptation](native_pixel_instruction_v61_result.md);
that adaptation also failed. Reusing its name or adding an adversarial or
reinforcement objective is not, by itself, a new solution.

The decision sequence remains capability-first:

1. Finish the already running V82 comparison without changing its cases,
   budgets, sampler or criteria. Retain every output. No concurrent second
   model job is needed.
2. Apply its original gates: metric-arm autonomous exact answers at least
   24/32, foreground F1 at least 0.98, and both improved over the matched
   control. Report absolute fit and the metric's relative effect separately.
   A control that fits while the metric fails to improve it rejects the
   metric benefit; it does not erase the control's measured fit.
3. If neither arm demonstrates adequate autonomous fit, reject scaling this
   recipe. Choose a bounded diagnostic that distinguishes failure hypotheses,
   using the existing correct-history and supplied-code controls. Preserve the
   earlier direct-versus-flow evidence; do not rerun the same failed treatment
   under a different name.
4. A training-fit pass only permits further evaluation. Reliable new-string
   copying and held-out prompt-dependent answers, including different questions
   about the same passage, remain necessary before claiming language learning.
   Copying alone cannot pass the language requirement.
5. Compare efficiency only at matched task quality, including all inherited
   model parameters, external pretraining and local training/runtime costs.

Checkpoint replay, reproducible execution, renderer preparation, manuscript
builds and attractive concept figures are useful engineering artifacts. None
substitutes for legible, correct answers generated from the prompt alone.
No usable ILM checkpoint or confirmed efficient learning algorithm is claimed.
