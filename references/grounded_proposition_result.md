# When a pictured answer credits the wrong person

A film programme says that 闻溪 wrote the screenplay of 《雨巷来信》 and asked 林遥 to direct it. Asked to credit the director and then the screenwriter, the model gives 《雨巷来信》的导演是林遥。《雨巷来信》的编剧是林遥。 After the names in the source are exchanged, both credits go to 请闻溪. The film has a legible title, the sentences have plausible syntax and the roles refer to different people on the page. The writer loses that distinction and copies the verb 请 into the second version of the name. Both returned images are preserved. [Input and outputs, Figure 92](../publication/ilm-nature/figures/Figure_92_grounded_proposition.png)

## What the writer learned

We prepared 1,700 training boards from Baidu DuIE 2.0 relation annotations. Each has two competing facts. Exchanging the names preserves the inventory of visible glyphs. We asked for each fact before and after the exchange. In the held-out portion, neither the subjects nor the source passages appeared in the added training set. These exclusions apply to our adaptation. Prior exposure in Chinese language pretraining is unknown. Baidu and the LUGE organizers distribute DuIE 2.0 for noncommercial research. [Official release](https://github.com/PaddlePaddle/PaddleNLP/issues/447) · [Corpus counts and hashes](evidence/grounded_proposition_20261119/corpus.json)

| Held-out question | Single-call ordered writer | Single-call, varied wording | Writer with a learned clause planner |
|---|---:|---:|---:|
| Both facts in order, practised combinations | 118/312 | 153/312 | 168/312 |
| Both facts in order, combinations withheld from compound training | 3/136 | 11/136 | 70/136 |
| All four replies correct on a withheld passage | 0/34 | 0/34 | 15/34 |

Withholding the combinations still allowed each role to occur on its own. On the 34 withheld passages the passage-paired gain from the planned system over the original ordered writer was 49.3 percentage points (95% bootstrap interval 34.6–64.7). It identified the correct physical spans of the subject and the ordered roles on 444/448 compound requests. Supplying the exact spans still gave 70/136 complete replies. Reversing the predicted role assignment gave zero on the same requests. In the first held-out passage it correctly exchanged the lyricist and singer of 《暖心家园》 in both orders. The other panel of Figure 92 shows the film error on different wording. [Planner comparison](evidence/grounded_proposition_20261119/planned_result.json) · [Passage-level uncertainty](evidence/grounded_proposition_20261119/board_intervals.json)

On the initial single-fact task, the genuinely paired fit answered both roles under both name assignments for 40 of 112 held-out passages using familiar questions. The equal-update fit with disrupted questions did so twice. Merely rephrasing the questions reduced the genuine fit to five passages. Learning the page–question association helped, but its expression in training mattered. [Genuine fit](evidence/grounded_proposition_20261119/matched_result.json) · [Disrupted fit](evidence/grounded_proposition_20261119/shuffled_result.json)

## Finding a name and writing a supported sentence

| System | Complete statement on 32 programme questions | Complete statement on 48 human questions | Human responses containing the reference name |
|---|---:|---:|---:|
| Planned writer | 14 | 2 | 26 |
| Separate writer trained with correct evidence, supplied with reader predictions | 15 | 3 | 27 |
| Same writer given the reference evidence | 28 | 3 | 28 |
| Separate writer trained with a competing name, supplied with reader predictions | 12 | 0 | 9 |
| Local Qwen3-8B, original text input | 0 | 21 | 45 |

The complete-statement scorer recognizes a limited set of formulations. On the 32 programme questions a reading of the original responses found that Qwen conveyed the requested fact on all 32, usually in different words. Two replies mentioned both participants and correctly assigned both roles. The planned writer supplied 14 complete correct credits. Two further replies began with evidence for the correct answer but contradicted it in their final sentence. This is an agent reading, without independent human adjudication. The original 48 human questions did not explicitly require a full sentence. With reader-selected evidence, 12 additional outputs exactly reproduced the direct human answer as a stopped bitmap sequence, bringing the combined count to 15/48. Supplying reference evidence gave 14 direct answers plus three full statements. [Direct-answer recount](evidence/grounded_proposition_20261119/direct_answer_scores.json) [All 32 programme responses](evidence/grounded_proposition_20261119/programme_readings.json) · [Descriptive scoring](evidence/grounded_proposition_20261119/programme_reading_audit.json) · [Qwen on original human questions](evidence/grounded_proposition_20261119/qwen_human.json)

The span reader returned the exact reference name on 21/29 simplified CMRC questions and 19/19 traditional DRCD questions. Supplying the reference name on all 48 still gave only three accepted statements. Outputs ranged from useful direct quotations to 丁善玺的导演是丁善玺。, which replaces the film title with the director. For the programme notes, the reader found only 15/32 correct spans, and supplying all the right spans instead raised the statement count from 15 to 28. These two cohorts expose different errors. On the programme notes, selecting the participant matters. On human questions the writer also needs to identify the subject, infer the relation in ordinary phrasing and decide how much to say. [Predicted and reference evidence](evidence/grounded_proposition_20261119/evidence_result.json)

The planned writer retained 190/192 historical captions with familiar prompts and 30/192 with rephrased prompts. It matched 10/571 open answers, all single spans, and 0/300 requiring a content glyph absent from the prompt. It can copy an unseen mark from the source and generate a printed glyph from a finite bank. Freely synthesizing a new historical form and general language acquisition from visual experience remain open.

## Reproducing the study

The released planner takes a passage PNG and a question PNG and returns an answer PNG. The two-sentence planner and the single-sentence evidence experiment are distinct checkpoints. Both use a pretrained Chinese generative body. The evidence variant additionally uses the transferred CKIP reader. No external language-model service runs during inference. Use the instructions bundled with each archive. The archives are stored in the local experiment directory and the versioned manuscript package under its models directory. [Planned writer usage](../models/grounded_proposition/README.md) · [Evidence writer usage](../models/grounded_proposition/EVIDENCE.md)

With the earlier visual checkpoints and study corpora installed, the core commands are

```sh
PYTHONPATH=. python scripts/grounded_proposition_corpus.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_grounded_propositions.py matched shuffled
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/extend_grounded_propositions.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/vary_grounded_questions.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_pictured_clause_planner.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/evaluate_grounded_propositions.py parent matched shuffled ordered unordered varied replay
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/evaluate_pictured_clause_planner.py
```

The planner fit starts from the varied checkpoint. The ordered and varied follow-up fits are defined in the linked protocols and use the scripts extend_grounded_propositions.py and vary_grounded_questions.py. The full fitted planner system has 86,726,729 parameter elements and 25,404,434 buffer elements. Its separate clause writer adaptation took 52.8 seconds and planner fitting 19.8 seconds on one 4090 D. The warmed model took a median 59 ms on 32 short programme questions, with 0.51 GiB peak GPU allocation. The eight archived examples reproduce the evaluated pixels and stop decisions on CPU. These costs exclude earlier training. The measured tasks and unequal answer quality cannot establish matched-quality efficiency against Qwen. [Ordered fit](grounded_proposition_order_protocol.md) · [Question variation](grounded_question_transfer_protocol.md) · [Planner training](pictured_clause_planner_protocol.md) · [Deployment measurements](evidence/grounded_proposition_20261119/release.json)
