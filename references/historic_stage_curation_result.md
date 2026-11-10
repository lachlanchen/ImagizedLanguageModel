# Correcting historical stage boundaries

The source parser had a concrete boundary bug: it searched forward from a
historical heading until finding a list, even after crossing the next heading.
An empty Oracle section could therefore inherit a Bronze, Seal or Liushutong
list. That copied the same physical glyph into a wrongly labelled stage.

The parser now stops at the next heading. Three regression tests cover empty
stages, unrelated intervening headings and populated sections. Four actual
cached source responses also match their declared counts after the fix:

| Character | Oracle | Bronze | Seal | Liushutong |
|---|---:|---:|---:|---:|
| 斯 | 0 | 2 | 1 | 16 |
| 性 | 0 | 1 | 1 | 6 |
| 言 | 21 | 11 | 1 | 33 |
| 中 | 34 | 70 | 1 | 31 |

The [cached-response check](evidence/hanziyuan_empty_stage_regression.json)
binds each original HTML response by hash. No new website requests were made.

The [derived inventory](evidence/historic_stage_curation_20261111.json)
retains **67,873 stage-consistent path-SVG records**, representing **65,133
distinct SVG byte hashes** and 5,553 modern character records:

| Source stage | Retained records | Modern characters with records |
|---|---:|---:|
| Oracle | 22,224 | 828 |
| Bronze | 18,565 | 1,437 |
| Seal | 5,161 | 5,161 |
| Liushutong | 21,923 | 3,450 |

All **16,753 wrong-stage records** have a byte-identical counterpart under the
correct source stage for the same character/label. They are quarantined in the
derived manifest, alongside sixteen non-SVG records. No glyph file is deleted,
copied or moved, and the shared legacy database remains unchanged. Every
retained SVG is content-hashed and XML/geometry checked. This is source-category
and structural validation, not independent verification of meanings, dates,
visual readability or archaeological interpretation.

There remain **2,701 identical-SVG groups shared across modern characters**.
These require grouped splitting and contextual/variant review before semantic
training. Curation does not force every shared form into a unique modern
identity and does not establish historical reading or generation ability.
The active CMRC writer experiment does not use this corpus; this bug therefore
does not explain its previously measured output failures.

The full local records and quarantine are in
`artifacts/historic_stage_checked_v1/`. Reproduce with the existing source
workspace and fresh output directories:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_historic_glyph_inventory.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/curate_historic_stage_inventory.py
PYTHONPATH=. python -m unittest discover -s tests -p test_hanziyuan_sections.py
```

These preparation steps call no neural model and make no network request.
