# The small pixel head fits every training block but not unseen questions

The unchanged 3,453,376-parameter head now generates **512/512 training first
blocks exactly**, including **256/256 Chinese and 256/256 English**. Every
generated pixel matches its owning reference. The head receives a detached
image-derived contextual state and its own preceding pixel patterns, with no
target image, target transcript, font lookup or answer retrieval at generation.
This establishes a conditional fitting capability, not general language
understanding or a complete PNG-input ILM.

The same checkpoint gives **0/128 exact Chinese and 0/64 exact English
development first blocks**. Both questions are exact in 0/64 Chinese article
pairs. It still fails both prospective development exactness criteria and
cannot qualify the full component or the broader ILM goal.

| Autonomous physical first blocks | 1,200-update head | 16,000-update head |
| --- | ---: | ---: |
| Chinese training exact | 0/256 | 256/256 |
| English training exact | 12/256 | 256/256 |
| Chinese development exact | 0/128 | 0/128 |
| English development exact | 0/64 | 0/64 |
| Chinese development foreground F1 | 33.498% | 40.040% |
| English development foreground F1 | 37.629% | 38.867% |

The original [shorter run](conditional_pixel_density_result.md) is preserved.
The [longer protocol](pixel_density_long_fit_protocol.md) restores its exact
untrained initialization and keeps the architecture, 512 examples,
normalization and first 75 example permutations unchanged. It executes
1,000 complete epochs, 16,000 logical updates and 512,000 retained example
draws. The cosine learning-rate decay also spans that longer schedule, so the
comparison does not isolate update count alone. No checkpoint is selected by
development quality. One exact update-three replay adds one executed update
and 32 draws without adding retained learning exposure.

Before this control, a complete CPU [correct-prefix diagnosis](evidence/pixel_prefix_diagnosis_result/report.json)
showed 76.831% Chinese and 90.796% English training-piece accuracy in the
shorter head. Even with correct preceding target pieces, complete training
exact counts remained 0/256 and 12/256. Its correct-prefix foreground F1 was
75.232% Chinese and 87.755% English. These assisted outputs are not autonomous
answers. They show that generated-history drift was not sufficient to explain
the shorter run's training failure. The longer fit establishes that the same
architecture and contextual states can represent those training targets; it
does not establish what is needed for generalization.

Matched whole-training-set NLL measurements at updates 0, 1,200, 4,000, 8,000
and 16,000 are retained in the [monitor](evidence/pixel_density_long_fit_result/monitor.json).
Final mean training NLL is 2.6094e-8 nats per eight-pixel piece, with 3.9094e-8
Chinese and 1.3095e-8 English. The likelihood conditions on true preceding
pieces, while all exactness counts above use autonomous generation. Very low
training loss and perfect fitting coexist with incorrect development images.

Chinese development question erasure reduces foreground F1 from 40.040% to
31.324%. The effect is 8.716 percentage points, with a 95% article-bootstrap
interval of [6.381, 11.447] points. English all-prompt erasure gives 25.984%
F1 versus 38.867% normal. These controls show input sensitivity without exact
answers. No independent semantic accuracy is measured, and the development
cohorts are reused rather than sealed.

Active head training takes **279.131 seconds (4.65 minutes)**, all 896 final
blocks take 12.721 seconds to generate, and the execution including checks
and saving takes 295.602 seconds. Peak allocated VRAM is **0.825 GiB** on one
RTX 4090 D. These costs exclude external pretraining, the earlier 8B reader
and its cache capture, preparation and audit. No new backbone is loaded.
This is not an efficiency comparison with a conventional language model at
matched useful quality. GPU1 remains untouched and the model process exits.

The registered causality/cache checks and actual model/optimizer replay pass.
The existing independent auditor verifies all 896 original PNGs against their
emitted bits, recomputes every physical score and checks source bindings.
All six complete native-pixel plates were actually inspected. All training
blocks match; development writing remains frequently distorted or incorrect.
No visual repair or manually assigned semantic score enters the results.

The complete [evidence archive](evidence/pixel_density_long_fit_result/result.json)
contains every original output, all case scores and training history. Training
plates cover [000–127](evidence/pixel_density_long_fit_result/plates/train_0000.png),
[128–255](evidence/pixel_density_long_fit_result/plates/train_0128.png),
[256–383](evidence/pixel_density_long_fit_result/plates/train_0256.png) and
[384–511](evidence/pixel_density_long_fit_result/plates/train_0384.png).
Complete [Chinese](evidence/pixel_density_long_fit_result/plates/development_zh_0000.png)
and [English](evidence/pixel_density_long_fit_result/plates/development_en_0000.png)
development plates include their unchanged erasure controls.

The result separates accurate emission for learned states from the unresolved
conditional generalization problem. The next integrated model must invoke the
reader on actual input PNGs, learn from complete answer-image histories, stop
appropriately and answer held-out prompts. Fitted first blocks cannot replace
those requirements. Historical forms, novel writing styles and general
independent language generation remain unqualified.

The current [supplement](../publication/ilm-nature/supplementary_information.pdf)
reports the architecture, likelihood, correct-prefix diagnosis and matched
shorter/longer fits in Note 56 (page 73). Figure S33 (page 74) shows the
editable method diagram, matched full-training NLL and fixed training/
development images. The main Results paragraph is on article page 14.
