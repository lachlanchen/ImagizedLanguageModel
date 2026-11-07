# Does recognition supervision improve actual generated writing?

The paired image writer recognizes only 10/32 held-out copying outputs exactly
with the separate CPU evaluator. The raw pages contain omissions and repetitions.
Velocity MSE alone does not explicitly penalize a misspelled string. This
experiment tests an additional training-only recognition loss; it does not
assume this explains all prior failures or establishes language understanding.

## Teacher admission, before student optimization

Use the public GOT-OCR-2.0 HF conversion, revision
`d3017ef2c2c1395888c8d635c5e0508bcb0ac78d`. Verify source hashes, use the
existing image-generation Python environment, and acquire no duplicate model
installation. The teacher is frozen. Its text vocabulary is permitted only
inside the training loss and the separately identified calibration process.
It must never enter the student's deployment path.

Calibrate on all 64 clean target pages from the existing strip copying cohort
(first 32 training examples and all 32 validation examples). No subset selection.
Use native plain OCR prompting, greedy decoding, at most 64 new tokens, and
the previous NFC/whitespace normalization. Preserve every raw transcript.
Admission requires at least 60/64 exact clean readings and at least 28/32
for each language; mean blank-page target NLL must exceed clean-page NLL by
at least 1 nat. Check finite nonzero image gradients for the first Chinese and
English training targets, frozen weights without gradients, and agreement of
the differentiable resize/normalization with the native fast processor
(mean absolute error <= .005, maximum <= .08). Record all checks, actual
parameters, time and GPU memory. Failure rejects this teacher/geometry recipe;
do not silently tune thresholds, crop text or drop difficult examples.

## Matched student test, conditional on teacher admission

Reuse the original pinned undistilled FLUX.2-klein-base-4B, native frozen VAE,
fixed generic condition, first 32 strip-image training pairs, rank 16,
512 updates, batch 1, seed 20268321, AdamW learning rate 1e-4 and decay .01,
gradient clipping 1, BF16 forward and FP32 adapter masters. Start fresh adapters.
Use exactly the existing uniform sigma/noise generator and example sampler.
Retain every training ID and sigma so the old MSE-only journal can be compared.

For native noisy target z_s = (1-s)y + s e and predicted velocity v, decode
the predicted clean image y_hat = z_s - s v using the exact native unpatchify
and VAE batch-normalization inverse. For s <= .5, add

    L = mean((v - (e-y))^2) + .05 * CE(R(decode(y_hat)), target string + EOS).

Otherwise use the original velocity loss. CE is teacher-forced, with only
answer positions scored, including EOS. Resize and normalize actual decoded
pixels differentiably; never render the label as the model's answer. Freezing
the teacher/VAE must not detach their input gradients. Retain the flow term:
recognition alone can reward adversarial marks. The coefficient and sigma gate
are fixed here before any student result is observed.

Run a two-update gradient/memory preflight before the 512-update experiment.
Keep one model job on GPU0; abort rather than exceed 22 GiB allocated memory.
If the teacher/gradient admission fails, stop this recipe without student
optimization and publish the failure, not a replacement successful subset.

## Observation and decision

Generate all 64 image-only answers in a fresh process with the same source,
50 native steps, guidance 1, 512x256 canvas and seeds from
`artifacts/image_edit_learning/full_strip_inputs.json`. That process loads no
teacher, labels, tokenizer or training examples. Score with the original
separate CPU evaluator after its retained reference calibration. Preserve all
raw PNGs, tensors and transcripts. Compare to the old MSE-only 6/32 seen and
10/32 validation observations. Manual review of every output is necessary;
OCR scores alone cannot admit an unreadable writer.

Writing admission remains 30/32 exact, legible requested strings alone in
each split, without repair, substitution or sample selection. A copying pass
would permit a separately registered semantic QA test, not establish a useful
ILM, independent language learning, arbitrary historical-script competence,
from-scratch training, or efficiency at matched LLM quality.

Primary implementations: [GOT-OCR2](https://github.com/Ucas-HaoranWei/GOT-OCR2.0),
[HF model](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf), and
[Transformers GOT API](https://huggingface.co/docs/transformers/model_doc/got_ocr2).
