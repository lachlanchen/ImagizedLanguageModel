# Selecting an object after reading the question

In 舉杯邀明月 the speaker raises a cup and invites the moon. The previous image reader committed to one object before it read the question. A new head scores every visible verb and its possible object after reading the question. On the same image it now selects 杯 for *raise* and 月 for *invite* and writes the corresponding English words. This previously seen line demonstrates a change in behavior. The measured transfer comes from separate held clauses. [Source pixels](evidence/questioned_events_v1/moon-page.png) · [Generated cup](evidence/questioned_events_v1/moon-raise.png) · [Generated moon](evidence/questioned_events_v1/moon-invite.png)

**Across 971 held clauses with multiple events, the new reader located every verb and object and drew every registered word on 247 clauses.** It completed 70 of 336 clauses from held works and 177 of 635 from held sections. The image writer could complete 741 clauses when given the correct positions. The comparison uses human dependency annotations in Kyoto UD release 2.16. Some recorded action names and object glosses differ from an idiomatic reading. [Kyoto treebank](https://universaldependencies.org/treebanks/lzh_kyoto/index.html) · [Frozen screen and cases](evidence/questioned_events_v1/held.json)

| Reader, original action names | All verb and object positions | All positions and complete words |
|---|---:|---:|
| One fixed pair per line | 0 | 0 |
| Frozen pair scores conditioned on the question | 64 | 51 |
| Exchanged object bindings | 16 | 11 |
| Learned verb, choose the following cell | 271 | 201 |
| Learned pair scores | 307 | 247 |

The learned reader completed 122 of 524 clauses containing a nonadjacent object. The following-cell rule wins on the held-work subset (81 versus 70); it relies on the learned reader to identify the verb. Neither the spelling of a correct word from a different source cell nor an unfinished output earns full credit. The published results include verb and object accuracy, supplied-verb object selection, supplied-object writing and a paired recount of original output pixels. [Results](evidence/questioned_events_v1/evaluation/result.json) · [Paired recount](evidence/questioned_events_v1/evaluation/audit.json)

### What happens when the action is described?

A second training comparison gave each event a second English view drawn from WordNet. In its matched control we exchanged only the described questions among eligible training examples and kept the literal event unchanged. The test pages and outputs below count complete bindings and stopped writing under both questions for every event.

| Questions | Pages | Literal training | Definition training | Exchanged definitions |
|---|---:|---:|---:|---:|
| Named actions | 971 | 247 | 257 | 209 |
| Dictionary descriptions | 118 | 0 | 2 | 2 |
| Newly composed natural questions | 112 | 2 | 5 | 1 |
| All complete questions previously unseen | 21 | 0 | 0 | 0 |
| Named actions on Pu Songling passages | 10 | 0 | 0 | 0 |

The pre-fit screen found that 16 of 106 complete natural questions could be produced by earlier training templates and vocabulary. The first preparation stopped before optimization. Both new arms avoided these exact strings. The original wording totals and the stricter 21-page measurement are reported separately. The candidate retained the literal and nonadjacent scores but tied the exchanged control on dictionary descriptions and completed no wholly unseen-question pages. The preregistered wording criterion therefore fails. The ten Pu extracts were screened against training pages and target windows. The writer could complete one of them when both object cells were supplied. [Registered wording test](questioned_event_wording_protocol.md) · [Results](evidence/questioned_events_v1/description_evaluation/result.json) · [English prompt collisions](evidence/questioned_events_v1/natural_question_overlap.json) · [Pu Songling source](https://www.gutenberg.org/ebooks/51828)

A post-fit probe restricted the candidates to the annotated events and required only correct event selection. On the 21 wholly unseen-question clauses the description-trained model selected all events on one clause under unrestricted search and three with the correct candidate pairs supplied. Giving the optical English encoder the correct English action names as alternatives produced four. With known object positions the writer could complete 16. Improving unrestricted spatial search alone is unlikely to remedy failures on these questions. The next comparison should first validate an English semantic teacher on reserved whole-question paraphrases, then distil it into the optical question encoder using separate training material. Use new source passages and new wording for the decisive test. [Scores with supplied candidates](evidence/questioned_events_v1/conditional_diagnosis.json)

### Derivation, training and reproduction

For each candidate verb the object maximizing the arc score can be cached. The question-dependent term is constant across that verb's candidate objects. Comparing the resulting per-verb maxima yields a joint maximizer. Lean checks this statement for integer scores. The deployed evaluator reads actual float-valued states and draws a stopped English word from the selected position. It expects a rendered, cell-aligned line and uses the inherited English glyph motor. Arbitrary script generation, unconstrained dialogue and efficient general book modeling require further evidence. [Lean proof](../formal/span_list/QuestionedEvents.lean) · [Model equations](../ilm/visual_lm/questioned_events.py)

Each head adds 65,427 parameters for a total of 3,592,031. Each fit used 102,400 sampled pages in 1,600 steps on one RTX 4090 D. Original literal-aligned optimization took 6.72 seconds following 6.23 seconds of shared state preparation. The original driver energy probe failed. A cold refit reconstructed the four weight files byte for byte and reproduced the saved decisions, output pixels and scores. During the refit the GPU energy counter recorded 629 J for literal state preparation and 614 J for the aligned literal fit. These figures exclude inherited model training, CPU work and evaluation. A matched-quality advantage over a general-purpose LLM remains unmeasured. [Reproduction record](evidence/questioned_events_v1/clean_refit.json) · [Run the models](../models/questioned_events/README.md)
