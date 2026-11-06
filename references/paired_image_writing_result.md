# Paired visual learning improves a few copies but fails reliability

The native 4B image generator completes 512 local low-rank updates on 32
copying examples on one RTX 4090 D. In the fixed eight-case screen, offline
OCR exact match rises from **0/4 to 1/4 seen strings** and **0/4 to 1/4
validation strings**. Both exact outputs are Chinese; one validation string
was absent from local optimization. This is limited visual-copy learning,
not question answering, language understanding or an efficient general ILM.

All eight cases are retained in their original order:
[training cases](evidence/paired_image_writing_result/comparison_0.png) and
[validation cases](evidence/paired_image_writing_result/comparison_1.png).
Each row shows the reference, unadapted base and adapted output. The
[complete export](evidence/paired_image_writing_result/result.json) includes
every original input and answer, every generated PNG, all 64 reference pages
and supplied-codec reconstructions, the update journal and original reports.
None of the generated writing is OCR-repaired, replaced with a font or
selected by quality. CPU export verifies each generated PNG against its
retained postprocessed decoder tensor.

The first four seen outputs contain three clear errors: an instruction
fragment mixed into the requested Chinese string, missing/repeated English
letters, and a phrase with omitted content. Even perfect outputs on every
remaining seen example could reach only 29/32, below the declared 30/32
criterion. The reliability screen therefore fails. Full 64-case autonomous
generation was not performed; eight samples cannot be reported as that
qualification. The original [protocol](paired_image_learning_protocol.md)
remains unchanged.

The offline recognizer uses fixed `chi_sim+eng`, PSM 6 and the existing
normalization. It recognizes 27/32 training and 28/32 validation reference
pages exactly, including mistakes in English spacing and Chinese forms.
Generated-font recognition remains uncertain; one English validation output
visually appears closer to its reference than its OCR transcript suggests.
The original transcript and image are retained rather than overridden.
The visible seen-case errors independently establish the failed criterion.

The source is the externally pretrained undistilled FLUX.2-klein-base-4B,
revision `a3b4f4849157f664bdbc776fd7453c2783562f4d`. The base transformer has
3,875,544,576 parameters; 24,358,912 additional adapter parameters are trained.
The deployed transformer contains 3,899,903,488 parameters plus an
84,046,115-parameter VAE. The same generic numeric condition is used for
every example. No changing caption, tokenizer, text encoder, separate target image,
OCR or font renderer enters deployed generation. Its shared Qwen3 condition
compiler has external pretraining and is not claimed identical to the stock
FLUX encoder.

| Retained invocation | Wall time (s) | Peak allocated GPU memory (GiB) |
| --- | ---: | ---: |
| 512-update training | 795.624 | 9.501 |
| Eight adapted autonomous images | 184.721 | 9.536 |

These timings exclude source acquisition, preparation, discarded setup
failures and separate inference/control stages; they are not total project
costs or evidence of training efficiency at matched language quality. Actual
VAE target reconstructions average 97.749% foreground F1, with a 94.559%
minimum. Those supplied-image controls do not establish conditional writing.

The observed substitutions, omissions and ordering errors remain the practical
bottleneck. A separately specified [padding comparison](paired_padding_protocol.md)
tests the measurable overhead of empty input space while preserving every
source pixel and all targets. It is not an established correction to these
errors. Natural QA optimization, question interventions, longer writing,
historical forms and broader language remain unproved.
