# Localize the compact visual-language failure

## Decision and scope

Both completed compact fits fail the fixed language evaluation. The wider
writer retains full tested output rank but gives no complete exact QA answers.
Preserve both checkpoints, source files, outputs and scores. The next work is
a diagnostic comparison using these frozen weights; no new fit is selected.

The measured failure is unreadable, repetitive writing. Its cause is not yet
localized. Separate the following questions before changing the model.

| Diagnostic | Held fixed | Intervention | Measurement and interpretation |
| --- | --- | --- | --- |
| Codec control | Actual target ink and fixed codec | Decode the encoded reference glyphs directly | Pixel/foreground fidelity bounds what this codec can express for the diagnostic samples; this supplies the answer and is not generation. |
| Visual access | Visible prompt, frozen body and image encoder | Compare learned grouping with reference boundaries/counts, and compare image updates with native training targets | Feedback error, count/boundary errors and contextual-state differences test access and alignment; none is a language score. |
| Conditional writing | Correct reference-image prefix/history and trained writer | Supply reference lengths, sample the next span, then compare with shuffled contexts using identical noise | Latent/pixel error and context sensitivity locate writing failure before autonomous history drift. Reference lengths/history are explicitly supplied. |
| Numerical integration | Same weights, context, output length and initial noise | Compare the registered eight Euler steps with a prospectively fixed finer integration | A change in actual glyph quality isolates sampler sensitivity; a lower velocity loss alone cannot establish it. |
| Generated history | Same prompt, seed, writer and output budget | Compare reference-image updates with actual emitted-image updates at matched boundaries | Divergence after a usable first span tests accumulated feedback error; failure before feedback cannot be attributed solely to drift. |

Native embeddings may be used only as a labeled diagnostic bypass of the
image encoder. They introduce information and a distribution change relative
to the deployed visual path. A successful bypass does not qualify visual
reading, and a failed bypass does not uniquely identify a writer defect.
Never merge supplied-history results with autonomous answer counts.

## Registration before execution

Bind the actual checkpoint, codec, source, case identities and original pixel
hashes in a machine-readable request before running model calls. Fix sample
selection, integration counts, seeds, metrics and runtime ceiling in advance.
Include seen training examples and separate development groups to distinguish
failure to fit from failure to transfer. Keep these diagnostic samples outside
any later untouched confirmation set. Retain every selected failure.

Implement the interventions in a separate diagnostic entry point. Preserve the
authenticated V1 and wider model sources and deployment loaders. Reuse the
existing corpus, teacher cache and foundation. Test exact context/target index
alignment and matched noise before interpreting the measurements.

Run serially under the project GPU mutex on GPU0 after the shared-workstation
admission checks. Record actual allocated memory, elapsed time, waits and
process ownership. The diagnostic request must declare its case/call budget
and deadline before launch; this plan is not a completed registration or an
executed experiment. Do not start a competing training job.

## Decision after measurement

Choose one intervention from the earliest measured failure. Register a matched
training comparison only after this localization. Re-evaluate full autonomous
answers and paired-question dependence after any repair, retaining the
unchanged failed control. Development improvements need fresh source groups
for confirmation. The full goal also requires historical and unencoded forms,
usable task quality and measured consumer-GPU costs.
