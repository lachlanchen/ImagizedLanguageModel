# Removing the VAE reference does not repair supplied-content writing

Removing the source-image VAE reference input leaves every supplied-content
page inexact in this fixed control. The original source images, visual-reader
inputs, supplied answer strings, prompts and seeds remain unchanged.

| Writer at guidance 4 | Full reference | Reference removed |
| --- | ---: | ---: |
| Released source, Chinese | 0/2 | 0/2 |
| Released source, English | 1/2 | 0/2 |
| Released source, combined | 1/4 | 0/4 |
| Adapted writer, Chinese | 0/2 | 0/2 |
| Adapted writer, English | 0/2 | 0/2 |
| Adapted writer, combined | 0/4 | 0/4 |

The four clean reference pages all pass the same fixed OCR rule. Both
reference-removed arms fail the registered four-of-four writing control.
These are repeated development cases, not independent benchmark observations.
The one previously exact English source output becomes misspelled; the
observed single-case change is not a population-level effect estimate.

The [protocol](native_reference_latent_ablation_protocol.md) and
[case registration](evidence/native_reference_latent_ablation_registration.json)
were fixed and pushed before generation. Two full-reference execution replays
and eight reference-removed pages are generated, in the registered order.
Eight complete original full-reference counterparts are reused without
regeneration. Every generation still deliberately receives the correct answer
text through the source tokenizer and lexical table. Target PNGs do not enter
the generator. This is an offline supplied-content component control, not an
independent image-language answering experiment.

The intervention sets only the native denoiser's `cond_hidden_states` argument
to `None`. It preserves the semantic input from the original image and text.
A subsequent hook observes the real argument immediately before the forward
pass. All 400 ablated denoiser calls have a source reference before the
intervention and no reference afterwards; all 100 replay calls retain it.
Every other forward argument remains the same object across the intervention.
No input is replaced by a blank page and no source weight is edited.

The [independent CPU audit](evidence/native_reference_latent_ablation_result/report.json)
checks actual decoder tensors against all ten original PNGs and verifies both
full-reference execution replays exactly. All eight reader-input tensors match
their original counterparts. In the first case of each arm, the actual initial
noise, timestep, semantic sequence, pooled condition and pre-intervention
reference-latent values also match the replay bit-for-bit. It recomputes all
paired OCR counts from unchanged raw transcripts and checks every saved hook
observation. Registered code, data, original outputs and preparation bindings
remain unchanged. The full attempt has 500 denoiser calls and 1,000 denoiser
batch elements, with zero optimizer updates.

Execution takes 336.887 seconds at 16.429 GiB peak allocated VRAM on one
RTX 4090 D, including loading and all ten new draws. The retained diagnostic
runtime has 6,862,609,631 pretrained parameters including the lexical modules
and unused output head. The original reader queries and the completed
1,536-update connector/writer adapters are used; the later failed query-fit
checkpoint is not loaded. These figures do not establish efficiency at
matched language quality.

All ten new complete pages are inspected at full resolution. The
[raw review](evidence/native_reference_latent_ablation_result/raw_review.json)
retains incorrect Chinese-like forms, English misspellings, added punctuation
and very small text. No glyph replacement, contrast repair, reroll or
quantitative manual accuracy is applied. The unchanged eight original
counterparts and four references retain their earlier actual-review evidence.
Complete [new pages](evidence/native_reference_latent_ablation_result/images/),
[paired full-reference pages](evidence/native_reference_latent_ablation_result/full_reference/),
[inputs](evidence/native_reference_latent_ablation_result/inputs/),
[references](evidence/native_reference_latent_ablation_result/references/),
[raw OCR](evidence/native_reference_latent_ablation_result/assessment.json)
and [archive hashes](evidence/native_reference_latent_ablation_result/provenance.json)
remain available. Source and OCR access were available during review; no
blinded assessment is claimed. The model and coordinator terminate and GPU0
is released; GPU1 is preserved.

This result does not support removing the VAE reference as a sufficient repair
for the tested writing path. It does not rule out interference as a
contributing factor, or learning a different reference-free architecture.
Removal also changes the appended image-patch attention geometry, and the
adapted writer was trained with the reference present. Prompt conditioning,
semantic alignment and writing fidelity remain unresolved.

The next model change must address conditional prediction and reliable writing
together. Repeating the unchanged query-only recipe, further sampler tuning
without a defined hypothesis, or treating a supplied-target reconstruction as
a generated answer is unsupported. The earlier
[V68 exact-history diagnosis](codec_language_v68_diagnosis_result.md) already
shows that a faithful writing codec alone does not fix conditional prediction.
The full objective remains useful, independent language generation directly
between writing images; this control does not achieve it.
