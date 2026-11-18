# A pictured Chinese writing test with answers outside the input

On 300 of 571 held-out questions, the human reference contains at least one alphanumeric glyph whose bitmap is absent from the question and the retrieved passage. Even a perfect selector with permission to reuse and reorder input cells could exactly reproduce at most 271 references. A system that returns one contiguous interval from either image could reproduce at most 177. These ceilings apply to the selected reference image. Other phrasings can answer a question correctly. [Independent recount](evidence/composed_chinese_answers_preparation_20261119.json)

| Reference image | Training questions | Development questions |
|---|---:|---:|
| Appears as one interval in the passage or question | 641 | 177 |
| Requires rearrangement, with all cells present in the inputs | 398 | 94 |
| Requires an absent alphanumeric cell | 1,526 | 300 |
| **Total** | **2,565** | **571** |

The preparation screened 40,000 original DuReader 2.0 training records and 10,000 official development records from its Search and Zhidao subsets. Retrieval compares the question with paragraph windows before consulting the answer. The targets are human answers of 8–48 cells. For the rearranged and absent-cell groups, a lexical overlap filter retains answers that share multiple runs with the evidence. This selects a short-answer subset with measurable physical copy limits. Its proportions do not estimate the prevalence of these tasks in DuReader. [Original corpus](https://github.com/baidu/DuReader/tree/master/DuReader-2.0) · [Preparation](../scripts/prepare_composed_chinese_answers.py)

An independent pass through the raw archive verified the rendered question, passage window and answer for every retained record. Training and development share zero complete question images, zero complete passage images and two distinct answer images. A deterministic reading of four training examples per group found eight directly supported answers, two partially supported answers, one supported sense of an ambiguous expression and one comparison involving different entities. The reading assessed whether the excerpt supported the answer. External factual accuracy remains unverified. [Audit code](../scripts/audit_composed_chinese_answers.py) · [Sample selection and readings](evidence/composed_chinese_answers_support_reading_20261119.json)

The archive and rendered examples remain local under `data/raw/dureader2/` and `data/composed_chinese_answers_v1/`. The archive includes the upstream noncommercial research or teaching license. To reproduce the pictured splits with the original archive in place, run the following from the repository root.

```bash
python -m scripts.prepare_composed_chinese_answers
python -m scripts.audit_composed_chinese_answers
```

This preparation establishes a writing test. A future writer should be scored on the completed answer, stopping, fidelity of the newly emitted glyphs and whether the reply answers the question using the supplied evidence. Reporting all three groups will show where improvements come from. No writer has yet been evaluated on these examples.
