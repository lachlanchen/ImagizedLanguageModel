# Reading the referent of a pictured word

A frozen Chinese writer produced **玛丽帮助了汤姆。** from the pictured sentence *It was Mary who helped Tom.* The matched local-head pipeline produced **汤姆帮助了汤姆。** Both used the same writer, output inventory and stopping rule. Only the English reader's supervised target changed. On a naturally occurring clause, *anyone who deceives us*, the new reader selected *anyone* and *us* correctly, but the writer produced **人都是我们。** Figure 1 reproduces all four Chinese outputs from the evaluation archive. [Paper](../publication/ilm-nature/manuscript.pdf) · [Editable figure](../publication/ilm-nature/figures/Visual_referent_answers.pptx) · [Raw images and per-case decisions](../publication/ilm-nature/evidence/visual_referents.zip)

## Finding both participants

Both 312,244-parameter models start from the same pictured head-word locator. They see ordered grayscale cells and a predicate mark. Each maximizes the sum of four learned endpoint scores over two disjoint, uninterrupted ink spans. The local model learns the grammatical heads. When an enhanced relative dependency points to a visible antecedent, the resolved model learns that head's location. Both fits see the same samples in the same order. This identifies a specific change to the learning signal. The model still requires a supplied event location and aligned print cells. [Protocol](visual_referent_learning_protocol.md) · [Training data](evidence/visual_referents_v1/corpus.json)

| Test clauses | Both roles correct, initial reader | Local-head fit | Resolved-head fit | Resolved gains / losses versus local |
|---|---:|---:|---:|---:|
| EWT and GUM, overt link, 126 | 0 | 0 | **69** | 69 / 0 |
| EWT and GUM, ordinary, 1,069 | 803 | 829 | **822** | 40 / 47 |
| GENTLE, overt link, 27 | 0 | 0 | **9** | 9 / 0 |
| GENTLE, ordinary, 303 | 228 | 237 | **227** | 8 / 18 |

The 69 linked successes were all active clauses. All three linked passives on the official test remained incorrect. Across the entire external set the new model found 236 pairs, one fewer than the local control. Thus the improvement on linked clauses coexists with a meaningful cost on ordinary external sentences. No held corpus crop matched a training crop from either the current training set or the original locator. Each arm used 1,600 updates on one RTX 4090 D and peaked at 395 MiB of allocated Torch memory. The resolved fit took 18.3 seconds after initialization. These figures exclude the original training of both reader and writer. [Paired counts](evidence/visual_referents_v1/outcomes.json) · [Image overlap and replay](evidence/visual_referents_v1/audit.json) · [Fit receipt](evidence/visual_referents_v1/fit-target.json)

## Writing the event in Chinese

The 1,688,480-parameter book-rehearsal writer was frozen in all conditions. Its output decisions select images in a fixed physical inventory, followed by its own stop. The combined deployed model has 2,000,724 learned parameters plus the print tables. The full-input control reads the complete sentence. Three other conditions receive actor, verb and undergoer in that order, separated by fixed spaces and followed by a period. A deterministic bitmap mapping adapts between the two known print faces. Each test gives the event location. No target response or target length enters generation. The annotated-span condition diagnoses the limits of extracting three heads.

| Source form | Full input | Local reader | Resolved reader | Annotated participants |
|---|---:|---:|---:|---:|
| Ordinary affirmative | 236 | 202 | 202 | 202 |
| Subject cleft | 17 | 0 | **180** | 202 |
| Object cleft | 0 | 0 | **110** | 202 |
| Reported subject cleft | 0 | 0 | **176** | 202 |

Each denominator is 236. A hit requires the complete canonical Chinese bitmap sequence and a natural stop. Exact matching can reject legitimate wording. For example, one answer uses 教师 where the reference uses 老师. Each form includes 118 pairs with the two participants reversed. The resolved reader gave both correct Chinese outputs in 69 subject-cleft, 31 object-cleft and 65 reported-cleft pairs. Both models had prior exposure to the component vocabulary. The controlled events had appeared in earlier evaluations. The results test transfer across wording, with restricted affirmative meanings. [Writing panel](visual_referent_writing_protocol.md) · [Pair audit](evidence/visual_referents_v1/audit.json)

On every linked, ASCII-renderable GENTLE clause within the writer's 96-cell window (eight), the resolved reader recovered both annotated participants in three. Its eight outputs all stopped. None conveyed the complete event in a single unblinded bilingual project assessment. None of the four conditions produced a canonical reference image. The reference descriptions were prepared before generation. Eight hand-assessed cases measure a failure in this sample and give a limited estimate of wider performance. [Individual judgments](evidence/visual_referents_v1/external-assessment.json)

## An information loss that more writer training cannot undo

For distinct target meanings with the same visual representation, a deterministic writer receives the same evidence and emits the same answer. At least one of the two answers must be wrong. The Lean theorem adds this lower bound over disjoint contrasts. On 236 emphatic affirmative and negative pairs that share their lexical verb, gold three-head views collided on all 236. The learned reader produced 194 collisions, imposing at least 194 errors in 472 requests even with ideal downstream interpretation. A differing representation on the remaining pairs does not establish correct polarity. [Checked Lean theorem](../formal/span_list/EventBottleneck.lean) · [Physical collision counts](evidence/visual_referents_v1/audit.json)

For subsequent model development, retain the unchanged pictured clause as well as the selected participants. Train a writer to use both views on genuine translated passages. Compare with a matched full-source writer and an exchanged-participant control. Reserve new texts, include polarity/role reversals, and judge autonomous answers on their full meaning. The present measurements establish neither broad independent translation nor a quality-matched compute advantage.

## Data and reproduction

EWT and GUM use their official Universal Dependencies 2.15 splits. GENTLE is an external test corpus with out-of-domain genres. The eight displayed and archived external passages retain the GENTLE treebank terms, CC BY-NC-SA 4.0, and original sentence identifiers. The corpus receipt pins all data files and identifies the GENTLE revision. [EWT and GUM attribution](evidence/natural_roles_attribution.md) · [GENTLE sources and attribution](https://github.com/UniversalDependencies/UD_English-GENTLE/tree/12cbbbbf9edef2036309cb3bb8e8db346cdad815) · [GENTLE license](https://github.com/UniversalDependencies/UD_English-GENTLE/blob/12cbbbbf9edef2036309cb3bb8e8db346cdad815/LICENSE.txt)

Use the project Python environment with PyTorch, Pillow, safetensors, OpenCC and NumPy. Prepare the original pictured-role corpus first. The full experiment runs with the following commands from the repository root.

```sh
export PYTHONPATH=.
python -m scripts.prepare_natural_pictured_roles
python -m scripts.prepare_visual_referents
python -m scripts.train_visual_referents
python -m scripts.evaluate_visual_referents
python -m scripts.audit_visual_referents
```

The audit compares all 3,808 archived answer image hashes, independently recounts retrieval, checks five actual writer replays against their original pixels and measures polarity collisions. The reader and writer commands can also process a single aligned image on CPU. [Inference commands and checkpoint](../models/visual_referents/README.md)
