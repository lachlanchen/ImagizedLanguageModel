# Does a visual writer learn meaning from a teacher's translation?

A 1,688,480-parameter visual writer independently translated three of 96 new pictured requests according to a masked, source-only bilingual reading. Its starting checkpoint translated two. A quantized 8B text model translated 88. The comparison does not establish a transfer advantage from distillation. The trained writer did learn to prefer and reproduce particular full translations on its training examples.

## Whole answers on previously unseen works

We reserved 24 passages from four novels and 24 passages from 24 talks. Both directions were evaluated. Both students read identical source images, chose successive bitmaps from the same visual inventory, and generated a stop. Qwen3-8B Q4 translated from source text. The final column measures a distinct test with eight grammatical variants per family.

| System | Registered whole meanings, 96 | Source-only whole meanings, 96 | Exact role families, 118 |
| --- | ---: | ---: | ---: |
| Starting visual writer | 0 | 2 | 111 |
| Corpus-trained visual writer | 1 | 1 | 107 |
| Teacher-trained visual writer | 0 | 3 | 113 |
| Quantized Qwen3-8B | 58 | 88 | — |
| Supplied reference | 95 | 80 | — |
| Shuffled reference | 0 | 0 | — |

The registered evaluation required acceptance in both candidate orders. A second review became necessary when the reference for “And this is one of the things we're doing” referred to starting at birth. Both trained students conveyed the pictured sentence, and the registered assessment credited only the corpus-trained answer. For the second reading, a Codex bilingual reviewer saw only the source and anonymously shuffled candidates. Identical wordings received one reading, with the original stop checked for each answer. The reviewer had previously seen 12 sampled cases, the first pair and the disputed example. No human adjudicator participated. The source-only document-bootstrap 95% intervals for the distilled gain were 0 to 3.13 percentage points over the start and 0 to 5.21 over the corpus control. The registered gain criterion failed. [Registered scores](evidence/teacher_translation_v1/results.json) · [Masked source-only readings](evidence/teacher_translation_v1/fresh-source-review.json)

The teacher-trained writer answered **她看了他一眼，却什么也没说。** with **“She looked at him and said nothing.”** The parent lost the negated action. On the first reserved literary source, the teacher-trained writer instead repeated “book” and failed to convey the sentence. None of the three credited output strings appeared as a full target among the 609,090 English-target or 610,645 Chinese-target distinct eligible pairs examined. This check includes candidate pools from earlier optical fits. Reuse of familiar phrases remains possible. [Original pictures](../publication/ilm-nature/figures/teacher_translation) · [Nearest-source and full-target audit](evidence/teacher_translation_v1/neighbors.json)

## What training changed

We sampled 4,096 training requests equally across domain and direction. The teacher produced 3,753 stopped, renderable translations. Two automated screens accepted 2,766. A masked source-only reading of 12 randomly sampled accepted and 12 rejected examples credited 11 and eight teacher translations, respectively. The screens can reject faithful phrasing. Within each stratum, integer selection maximized the number retained with identical total supervised target length in both arms. This yielded 2,614 requests, of which 2,563 had distinct target wordings. [Training protocol](complete_teacher_translation_protocol.md) · [Label review](evidence/teacher_translation_v1/targets-blind-review.json) · [Selection counts](evidence/teacher_translation_v1/selection.json)

Each student continued the same trained checkpoint with the same batches, replay, 2,208 updates, 7,424,220 supervised positions and 210,840 stops. The two twelve-pass fits took 27.4 and 27.7 seconds on one RTX 4090 D. Teacher generation and screening took 800.5 additional seconds. These are incremental costs on top of student and teacher pretraining. Development selection chose pass four for the corpus writer and pass two for the distilled writer. On the older 64 encyclopaedic requests neither student earned a registered semantic credit. Qwen earned 47. On the 96 fitted diagnostic requests the registered assessment credited 11 corpus answers and two distilled answers.

Across 2,563 alternative target pairs the mean complete log odds shifted 35.36 nats toward the teacher. The first differing glyph contributed 3.75 nats and subsequent decisions contributed 31.61. The Lean theorem proves the decomposition for additive scores over a common causal prefix, including the terminal decision. Its residual was zero for every measured pair using integer micro-nats. At pass twelve the distilled writer freely reproduced 1,117 of 2,614 teacher targets, compared with 154 at its selected pass. Neither last-pass writer reproduced any of the 96 fresh references exactly. This locates the effect in learning fitted wordings and leaves broad semantic generalization unresolved. [Lean proof](../formal/span_list/TeacherSurfaceFork.lean) · [Full accounting](evidence/teacher_translation_v1/forks.json)

## Next comparison

Vary the number of distinct source propositions at fixed glyph supervision, role rehearsal, starting weights and decoding. Reserve complete books before sampling, check both sides for overlap and grade fidelity to the source alongside the reference-informed score. Keep the complete images and all stop decisions. Additional epochs on the same translations improved reproduction of fitted wording. The new comparison tests whether exposure to more varied language improves meaning on unseen works.

The prepared ZhJpBook corpus contains **28,342** distinct aligned training pairs from 21 additional literary works, with 240 development pairs from two other works and 480 test pairs from four more. The preparation retains the document, chunk, paragraph and unit for each pair and checks glyph repertoire in both directions. These are candidate alignments, with semantic quality still to be assessed. Training in the experiment above used the separately screened 2,614 requests. [Corpus inventory and input hashes](evidence/visual_library_v1.json) · [Preparation code](../scripts/prepare_visual_library.py)
