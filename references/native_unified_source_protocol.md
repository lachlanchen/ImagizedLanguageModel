# Native unified source capabilities: fixed diagnostic protocol

Specified before GPU source-model calls, 7 November 2026. The previous goal
turn made progress by completing and retaining a failed 64-case copying
observation and updating the paper. No useful ILM checkpoint is qualified.

Test the existing Show-o2 HQ model as an initialization candidate, not a new
architecture or a working student. Pin official source commit
`45a5a2de01d1ebd10cd5864d29310a76476cdf23`, model revision
`d3a220ec55feaacbdfcb053847edee14edd4e69a` and actual model/VAE SHA256.
Code and model cards identify Apache-2.0; include component provenance.
Do not download redundant Qwen/SigLIP weights: their retained weights are
inside the complete checkpoint. Construct placeholders from pinned configs
and load every retained tensor strictly. Record actual complete parameters
and memory, rather than treating the 1.5B language-body label as deployment size.

Use the first four training copies, first four validation copies and first
four validation QA records in the existing `pages_v2/pairs.jsonl` order.
Copy inputs use the previously verified 1024-by-256 strip; every removed pixel
is white and all source ink coordinates are retained. QA inputs retain the
full original 1024-square page. No centre crop, answer-aware crop or image
resizing is allowed. The native semantic encoder's existing positional
interpolation handles the actual latent geometry. Record the preprocessing
and preserve input/latent hashes. These are reused exploratory cases, not a
sealed test or an independent-seed replication.

First run a diagnostic source-text readout on all twelve input images with
the same generic instruction to follow the request in the image and output
only the answer. Separately ask for transcription of the eight copy images,
using one fixed transcription instruction. No reference answer enters either
readout call. Use FP32, seeded native VAE posterior sampling and the original
native `mmu_generate`, top-k one, temperature one, with maximum 96 decisions
for answering and 128 for copy transcription. Use seed 20268521+i per input;
reset seed 20268621+i for each source readout. Keep the original EOS decisions,
including truncation. Reference strings are opened only by separate scoring.
Retain all outputs and codec input reconstructions. The source helper retains
its text vocabulary and is not a deployed ILM or a student language result.

Test native writing separately by explicitly supplying the eight copy target
strings to the source text-to-image path. This is an assisted rendering
control. Use the unchanged template: `A flat scan of a white page with a
single line of black text: "{string}". Plain readable print, generous white
margins, no decoration, no other text.` Use the native 512-square HQ layout,
50 time points, Euler ODE, shift factor 3 and guidance 5, with seeds
20268721+i. Retain complete raw images and postprocessed tensors. No font
substitution or OCR repair. Score with the already calibrated separate OCR
and inspect every output; compare expected text, not exact font pixels.

Only a later image-question-to-image-answer runtime with no changing language
lookup/output path can qualify as the student. Neither text-helper answers
nor text-supplied writing controls count toward that goal. A source failure
rejects a claim of inherited competence and requires a diagnosed learning
experiment, rather than blind scale-up. No local optimizer updates occur in
this diagnostic. Keep the full ILM goal active and unachieved.

The fixed answering instruction is `Follow the request written in the image.
Output only the requested answer, with no explanation.` The fixed transcription
instruction is `Transcribe all the text in the image, preserving line order.
Output only the transcription.` These generic instructions are source-helper
controls, not changing question captions.

Configuration-only construction under meta can temporarily split tied Python
Parameter objects. After strict loading, check that the actual checkpoint's
input/output table values agree before restoring the native declared tying.
For 512-square generation, set the derived nonpersistent position-index buffer
to the actual 32-by-32 latent patch grid, selecting the upstream positional
interpolation branch. The saved HQ config defaults to 27-by-27. This geometry
hydration changes no learned tensor; record it rather than claiming the
unmodified example script handles every resolution. Native forward and ODE
functions remain unchanged. The tuple-only timm imports use torch's integer/
tuple `_pair` helper to avoid installing unused torchvision augmentations.
