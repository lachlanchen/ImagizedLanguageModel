# Current direction: useful visual language generation remains unproved

Assessment date: 7 November 2026. The completed experiments do not establish
a useful ILM. The image-input/image-output research goal remains open; the
current learning recipe has not earned a scale-up. The completed V82 matched
comparison fails all four original repair criteria and is rejected for scaling.

The later [full-page curriculum](native_visual_alignment_result.md) improves
requested copying to 29/32 but leaves Chinese and English QA at 0/32 on the
original 1024-square pages. A subsequent
[matched geometry check](native_reader_geometry_result.md) changes that source
interpretation: the same frozen pretrained reader recovers 11/32 exact English
answers at the official 432-square interface, without training. Nine of those
correct answers disappear without the passage. Existing adapters give 5/32
each at 432, below the source. This identifies an effective interface correction
on this cohort and rejects the learned adapters as a demonstrated repair.
It does not isolate the geometric mechanism, qualify Chinese native-scale
reading or establish direct image answers. A calibrated source baseline must
precede another optimizer. The historical failures below remain unchanged.

The latest native-image [copying observation](paired_padding_result.md) also fails: 6/32 training and 10/32 validation OCR-exact strings after 512 updates, with visible omissions and duplicated writing. It has no natural-QA optimization. Reducing blank input space lowers training invocation time from 795.624 to 287.182 seconds without qualifying reliable writing. The failure concerns content as well as appearance.

The latest completed [readout comparison](decoder_metric_v82_result.md)
provides three distinct measurements on the same 32 seen training questions.
The metric arm is worse than the continued control (85.984% autonomous F1):

| Path | Metric-arm result | What it tests |
| --- | --- | --- |
| Autonomous answer, generated-pixel feedback | 0/32 pixel-exact; 70.009% foreground F1 | Actual complete image generation |
| Correct preceding answer codes and known length | 0/32 pixel-exact; 75.116% foreground F1 | Conditional prediction with answer-history assistance |
| Supplied target codes through the decoder | 28/32 pixel-exact; 99.9953% foreground F1 | Physical representation and decoding of known answers |

Pixel equality is stricter than readability, and foreground F1 is not semantic
accuracy. The [complete raw galleries](evidence/decoder_metric_v82_result/README.md)
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

The [PNG-only position diagnosis](evidence/decoder_metric_v82_position_diagnosis.json)
reproduces all six final summaries from the public pixels. Control writing
falls from 99.467% F1 on the first block to 76.237% on the remaining stream;
the metric arm falls from 97.018% to 47.493%. Remaining-stream averages cover
the 28 longer references; first-block averages cover all 32. Correct history
raises the longer tails to 87.724% and 56.682%, respectively, still short of
clean writing. This is a measured prediction failure, not evidence that a
particular new objective will fix it.

The decision is to stop scaling this repair. More data, a new loss coefficient,
an image-generator name or another publication figure does not repair an
unproved conditional learning path. The prepared
[image-generator baseline](image_edit_baseline_result.md) now completes all
40 autonomous normal/blank outputs in 91.656 seconds at 8.943 GiB peak
allocated memory. It fails all 12 copying requests and eight questions by
offline OCR exact match; reference calibration is 18/20. All outputs were
inspected and retained: copied instructions/passages and corrupted text do
not constitute the requested answers. There are no local optimizer updates.
The result rejects zero-shot use of this composition. The subsequent
[paired-writing run](paired_image_writing_result.md) completes 512 updates on
32 copies: the fixed screen improves from 0/8 to 2/8 OCR exact matches, but
three clear seen-case errors already rule out its 30/32 training criterion.
This demonstrates limited copying improvement, not useful language learning.
Natural QA has not been optimized in that model. A separate
[lossless-padding comparison](paired_padding_result.md) completes the full
64-case strip observation and fails reliability. All ink and target pixels
are preserved, but frozen-VAE input features change. It is not a fidelity repair.

A replacement must be assessed in this order:

1. Demonstrate accurate, autonomous visual writing from the prompt alone.
   Keep supplied-code and correct-history controls to distinguish physical
   rendering from conditional prediction. Preserve the earlier
   direct-versus-flow evidence rather than relabelling it as a new treatment.
2. Demonstrate readable new-string copying, including changed strings and
   lengths. Assess the actual unmodified output, not only latent loss or an
   aggregate image score. Passing this establishes copying, not semantics.
3. Demonstrate held-out prompt-dependent answers and both questions about the
   same passage. Separate answer correctness from legibility. A new-string
   copy pass or a training-fit pass cannot qualify general language learning.
4. Compare efficiency only at matched task quality, including all inherited
   model parameters, external pretraining and local training/runtime costs.

Checkpoint replay, reproducible execution, renderer preparation, manuscript
builds and attractive concept figures are useful engineering artifacts. None
substitutes for legible, correct answers generated from the prompt alone.
No usable ILM checkpoint or confirmed efficient learning algorithm is claimed.

A [completed native unified source diagnostic](native_unified_source_result.md)
now tests that initialization: image prompts give 0/12 canonical exact source
answers, while original text gives 5/12, with all eight copy contents present
and extra quotation marks on four. Supplied-input codec images remain readable.
This motivates a [visual-access learning experiment](native_visual_access_learning_protocol.md),
whose real two-update preflight passes at 6.899 GiB. Its
[completed 4,096-update result](native_visual_access_result.md) gives 27/32 exact
held-out copies through the auxiliary text head, but 0/32 correct natural
answers. Fifteen of sixteen paired questions receive the same answer; erasure
leaves 29/32 outputs unchanged. This visual-adapter recipe fails language
selection and does not admit image-writing adaptation or scale-up. It is not
a deployed image-answer model. The [full direction](native_visual_language_learning_direction.md)
still requires direct continuous image output and vocabulary-free deployment.
