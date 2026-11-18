# A pictured fact and a historical form in one answer

Two illustrated Hanziyuan entries share a card. Each entry carries a headword, a source-listed English gloss, a pronunciation and two historical forms. The question names a headword and asks for one fact and one stage. An answer image combines the headword, requested fact, stage and original historical pixels in a modern Chinese sentence. Across eight questions per card, changing the headword, field or stage changes the answer. The fields appear in independently shuffled order within each entry.

| Partition | Cards | Character families | Distinct complete answers | Additional question paraphrases |
|---|---:|---:|---:|---:|
| Training | 1,115 | 2,230 | 8,920 | 0 |
| Development | 142 | 284 | 1,136 | 1,136 |

The local cache supplies the records and original bitmaps. A separate recount located each requested field in the indicated source entry and verified every one of the 10,056 complete answer streams against the source fact and a historical raster digest. Exact historical raster, family and entry overlap across the two partitions is zero. The previous historical plate study used the same family partition. The new task and questions differ, while some individual shapes may have appeared in those prior evaluations. Meanings are the source's glosses and have not received independent philological review. [Preparation and recount](evidence/pictured_entry_answers_preparation_20261119.json) · [Source metadata](historic_entry_metadata_result.md)

Training, development and paraphrase views use 32-pixel binary cells. Each split has its own bitmap array. Integer offsets in the packed files locate pixels for storage. A visual learner must receive the gathered image arrays. The local index holds source identifiers and provenance for evaluation. Chinese answer frames and question wording are authored. Exact complete answers depend on evidence selected from both the question and the plate. An explicit field parser is a strong comparison because the field labels and target sentence structure are regular. This task measures grounded assembly of an original glyph and a lexical fact. Success alone would leave independent etymology, free translation and book-length writing open.

To prepare the files from the verified local historical cache and check them, run these commands in the repository. The output directory must be new.

```bash
python -m scripts.prepare_pictured_entry_answers --output data/pictured_entry_answers_v1
python -m scripts.audit_pictured_entry_answers --data data/pictured_entry_answers_v1
```

The intended comparison trains on the ordinary question phrasing. Its two development views use identical cards and answer images, so the cost of the new paraphrase can be measured separately from transfer to a new character family. Evaluate complete answers, exact historical ink, factual text, learned stopping and consistency after changing each requested field. Replacing the question or the plate independently measures input use. No training on this corpus was included in the Figure 69 experiment.
