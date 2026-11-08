# Recognition supervision does not repair autonomous copying

The completed matched experiment gives **6/32 seen and 9/32 held-out strings
exactly recognized by the separate CPU evaluator**, compared with 6/32 and
10/32 for the previous velocity-MSE-only writer. The clean references score
27/32 and 28/32. OCR underrecognizes some legible generated writing, so these
are OCR observations, not manual reading accuracy. Review of the complete
raw galleries independently finds more than two obvious failures per split.
The declared 30/32 exact, legible requested strings alone per split fails.
This checkpoint remains unsuitable as a usable ILM.

Every case appears in its original order, reference above actual output:
[training 00–15](evidence/recognition_guided_writing_result/all_cases_00_15.png),
[training 16–31](evidence/recognition_guided_writing_result/all_cases_16_31.png),
[validation 00–15](evidence/recognition_guided_writing_result/all_cases_32_47.png),
[validation 16–31](evidence/recognition_guided_writing_result/all_cases_48_63.png).
The [complete export](evidence/recognition_guided_writing_result/result.json)
contains all 64 inputs, references, answers, transcripts, source reports and
the 512-update journal. Every exported answer equals the quantized retained
neural decoder tensor. No OCR repair, font substitution or selected reroll is
used. This authenticates saved pixels; it is not an independent neural replay.

## Controlled learning comparison

Both runs use the pinned externally pretrained undistilled
FLUX.2-klein-base-4B, frozen native VAE, the same fixed generic condition,
fresh rank-16 adapters, first 32 copying pairs, 512 updates, batch one,
AdamW learning rate 1e-4, decay .01, clipping one and seed 20268321.
All 512 sampled IDs and noise times match the retained control exactly.
The preflight also reproduces the original native flow term bit-exactly;
its first update's flow loss equals the old control's first loss.

The treatment adds .05 times frozen GOT-OCR teacher-forced answer CE, including
EOS, when sigma <= .5. It scores actual VAE-decoded predicted-clean pixels:
`y_hat = z_sigma - sigma * predicted_velocity`. It changes neither the
training references nor the deployed image-only interface. This is a standard
auxiliary recognition objective, not a new language-learning theorem.

| Measurement | Original flow loss | Flow plus recognition |
| --- | ---: | ---: |
| Seen copying OCR exact | 6/32 | 6/32 |
| Held-out copying OCR exact | 10/32 | 9/32 |
| Training invocation time (s) | 287.182 | 374.748 |
| Peak allocated training memory (GiB) | 8.437 | 12.268 |
| Locally trainable adapter parameters | 24,358,912 | 24,358,912 |

This is one run per arm on an exploratory cohort, without independent-seed
replication. A one-case difference is not a general causal claim that all
recognition supervision is ineffective. The measured recipe fails its writing
requirement and provides no efficiency advantage at useful language quality.
These times exclude acquisition and separate calibration/inference jobs.

## The supervision executes correctly but does not qualify the writer

The training-only GOT conversion contains 560,528,640 parameters. Complete
clean-target calibration gives 62/64 exact readings (Chinese 32/32, English
30/32); blank-page target NLL exceeds clean-page NLL by 9.785251 nats on average.
Input gradients are finite and nonzero for both inspected languages, with no
teacher parameter gradients. The original calibration fails preprocessing
agreement. The [recorded correction](recognition_resize_correction.md) preserves
Pillow's separable clipping without lowering any gate; its repeat passes all
original criteria. Both complete reports remain in the export.

The two-update student preflight verifies a nonzero recognition gradient all
the way to native velocity predictions, nonzero gradients to the input image,
changed adapters and frozen teacher/VAE parameters. The full run applies the
recognition term on 255/512 updates and changes all 206 adapter tensors.
Mean recognition CE over those noisy-target-assisted updates is 0.031519 nats.
That low value does not become reliable autonomous writing. During training,
the recognition branch receives a prediction made from a noisy correct answer;
free inference starts from independent noise. This observed gap motivates a
further diagnosis of that supervision regime; it does not isolate the cause
from data size, optimization, source conditioning or the architecture.

All 64 fresh generations use the same input-only index and seeds as the
earlier full cohort, 50 native steps, guidance one and a 512x256 output canvas.
Their invocation takes 513.884 seconds with 8.031 GiB peak allocation.
It loads no teacher, tokenizer, varying text condition or reference answers.
The deployed transformer still has 3,899,903,488 parameters including adapters,
plus the 84,046,115-parameter VAE; this is not a 24-million-parameter model
trained from scratch. The fixed condition's external encoder provenance is
unchanged and disclosed in the [earlier result](paired_padding_result.md).

## Failure is present in the generated writing

Training case 004 omits characters; case 012 adds a second line copied from
the instruction; case 015 misspells the phrase and adds another line. Validation
case 000 adds marks, case 002 repeats a character, case 025 repeats a letter,
and case 026 duplicates its line. These are visible content failures, not
merely the evaluator misreading a font. Several other cases are readable and
correct; they support partial copying only.

This run contains no natural-QA optimization or evaluation. It establishes
neither semantic understanding, independent general-language learning,
historical-script generalization nor parity with a language model. Scaling
this unchanged recipe is not admitted. The next necessary evidence is correct
autonomous writing on new inputs followed by separately evaluated meaningful
answers; lower denoising loss or cleaner-looking characters cannot replace it.

Reproduce with the [execution guide](../docs/recognition-guided-writing.md),
the [original protocol](recognition_guided_writing_protocol.md) and
`scripts/export_recognition_guided_writing.py`.
