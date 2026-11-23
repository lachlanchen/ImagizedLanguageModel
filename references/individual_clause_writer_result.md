# What a visual writer learns from a single action question

On a pictured credit with two people, aligning the requested action with the correct name raised complete responses across a role exchange from 19 to 175 of 256 comparisons. On twelve new two-person stories the same training gave zero complete comparisons in either question language. The writer learned a repeatable relation in credits. It still confuses people in ordinary prose, including when every required glyph is available.

| Evaluation | Common parent | Independent labels | Aligned clauses |
|---|---:|---:|---:|
| Edited credits, both roles on both versions | 19/256 | 19/256 | 175/256 |
| Earlier stories, both roles on both versions | 0/24 | 0/24 | 1/24 |
| New stories, both roles on both versions | 0/24 | 0/24 | 0/24 |
| First 96 novel passages, both questions | 8/192 | 5/192 | 8/192 |
| Other 253 novel passages, both questions | 17/506 | 19/506 | 26/506 |
| Two supported statements on new stories | 0/48 | 5/48 | 3/48 |

The first five rows require each generated clause to stop, include the requested name or annotated excerpt, and omit the other. They pool the Chinese and English requests. The last row required accurate actions as well as names and order. One model-based reader assessed all 48 responses with arm identities hidden. Local text Qwen3-8B passed 38, including two semantically correct answers written in English despite the Chinese output instruction. The parent using the original joint question passed six. The aligned writer got none of the twelve stories right on all four readings. [Full blinded readings](evidence/action_binding_20261124/reviewed_answers.json) · [Scene-bootstrap differences](evidence/action_binding_20261124/paired_transfer.json)

On the ferry story, the writer correctly answered an English question pair in Chinese after the two workers switched jobs: “看水位的是田夏。傅岩收船票。” On the puppet story, it assigned both jobs to 宁皓, then both to 费音 when the names exchanged roles. Figure 111 shows both original strips. The training-label audit confirms identical ordered inputs, target frequencies and common replay in both fits. The additional 253 novel passages yielded a 1.8-percentage-point difference relative to the parent, with a paired 95% scene-bootstrap interval of −0.2 to 3.8 points. [Training audit](evidence/action_binding_20261124/exposure.json) · [Original figure](../publication/ilm-nature/figures/Figure_111_individual_clause_writer.png)

## Why legible answers can still be wrong

Write the two-name log odds as a two-by-two table, with rows for page version and columns for action. Adding a constant bias for either page or either question cancels in the signed interaction $D_{00}-D_{01}-D_{10}+D_{11}$. Correct assignment across all four cells requires the interaction to overcome the worst competing bias. Lean checks the exact sign conditions and the smallest uniform signed increment that attains a given decision margin in common integer units. In the earlier writer, the interaction was never the largest component in any of 24 story comparisons. These results give a test of the scores learned by a network. [Lean proof](../formal/span_list/ActionPageInteraction.lean) · [Observed decomposition](evidence/action_binding_20261124/name_odds.json)

Even with two complete answer images supplied for scoring, the aligned writer preferred the supported clause to the identical predicate with the other name on 975/1,024 credit questions and 53/96 new-story questions. The untouched parent also scored 53/96 on the new stories. All alternatives were writable. The new fit thus transferred little of its assignment advantage to the narrative test even when the decoder was spared the task of composing a reply. [Stopped-clause probabilities](evidence/action_binding_20261124/clause_odds.json)

## Four historical forms of 言

The approach can return exact off-font shapes when they are visible. The earlier four-prompt card reused a training card containing an oracle and a bronze exemplar for **each of 言 and 征**. We tested a separate card containing both characters in four labelled categories. For 言 the selected labels were J04903, B02975, S01648 and L06958. Liushutong identifies a collection. The order in this test is a requested catalogue order.

On a black-on-white page, none of eight single prompts returned all four forms. Splitting the question in two while retaining the full page also gave zero of eight. Splitting the source by the requested categories gave two of eight, both in Chinese, one in each requested order. The corresponding counts with the legacy raster polarity were zero, one and two. This tests exact reproduction of supplied images. The system has yet to give a reliable account of their historical development. [Source selection](yan_four_source_categories.md) · [All outputs and hashes](evidence/action_binding_20261124/yan_four_forms.json)

## Single-GPU cost and use

Both fits updated 28.6 million parameters of the 87.1-million-parameter writer with 76,052 presentations, 3,848 updates and 903,050 target positions. Both used the same 7,092 bilingual single-question book examples drawn from 1,773 paired question sets. A 32-clause spot check found two unsupported teacher labels. The aligned fit used 139.7 seconds of optimization, peaked at 4.11 GiB of allocated CUDA memory and used 35.3 kJ over the measured fit call. Preparing the additional book targets used 554 seconds of local Qwen3-4B service time. These are incremental costs above the existing trained parent. [Fit](evidence/action_binding_20261124/aligned_fit.json) · [Teacher reading](evidence/action_binding_20261124/book_clause_reading.json)

On 48 warm two-question requests, the visual writer reproduced every scored image and stop, with 84 ms median request latency and 482 J total device energy. Local text Qwen3-8B took 326 ms median and 3,360 J. The images were prepared in advance and the energy excludes the host. The measured advantage at 3/48 supported replies does not establish a saving at the text model's 38/48 accuracy. [Vision measurements](evidence/action_binding_20261124/warm_inference.json) · [Text measurements](evidence/action_binding_20261124/text_control.json) · [Offline image inference and training](../models/individual_clause_writer/README.md)

## Next comparison

Preserve this writer for the credit and catalogue tasks. A new grounding test should require a visual reader to choose a complete pictured statement supported by a natural passage before asking a writer to compose it. In particular, give the reader two fluent clauses that differ only in the participant. Train on diverse narrative actions with hard negatives and hold out entire predicates and works. Compare the same reader with exchanged labels at equal exposure and freeze new stories before fitting. Only a gain on novel actions across both page versions would justify connecting this reader to the writer for another free-generation trial. This isolates a measurable failure in the current scores. A related pixel-only study found a benefit from writing intermediate visual steps on symbolic problems; whether an evidence-selection step helps with these Chinese narratives is an empirical question. [Chu and colleagues](https://arxiv.org/abs/2611.22820)

In a follow-up with all weights frozen we swapped the written names consistently in the prompted page and in both supplied answers. Averaging scores before and after this exchange cancels a fixed preference for either name. It left the aligned writer at one complete choice comparison in 24 new-story comparisons and reduced correct individual choices from 53 to 48 of 96. [Derivation and supplemental results](action_renaming_probe.md)
