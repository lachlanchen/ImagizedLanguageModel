# Existing historical glyph corpus

The historical source database is available in the legacy workspace at
`../incoder/data/historic/etymology.sqlite3`; its assets are under
`../incoder/data/historic/glyphs/`. The current model repository does not need
a second copy. The [read-only audit](evidence/historic_glyph_inventory_20261111.json)
checks the existing database and file paths without recrawling the website or
changing that source workspace.

The database has 9,055 character records and 84,642 glyph records. Of those
characters, **5,553 have glyph records**; a database character row alone does
not establish the presence of a historical form. Every recorded file exists
inside the declared asset root.

| Source stage | Recorded glyph files |
|---|---:|
| Oracle | 32,216 |
| Bronze | 23,500 |
| Seal | 6,996 |
| Liushutong | 21,923 |
| Unknown | 7 |

There are 84,626 SVGs and sixteen other files. All sixteen non-SVG records
belong to 中; its 136 SVG records agree with the previously demonstrated
source extraction count. The audit does not treat the extra images as
historical glyph training targets. A deterministic sample of 339 files has
no sampled SVG XML parse failures or external/active SVG references; unsampled
content and individual historical meanings have not been verified.

**3,300 stage/source-label groups occur under multiple modern characters.**
These can reflect variants, shared forms or other source relationships; the
count alone does not prove that all such records are errors. They should not
be split independently across training and evaluation, or blindly forced to
different unary semantic targets. Future use needs source/variant checks and
grouping by the actual physical form. A shared image can require linguistic
context to disambiguate its meaning.

The current natural-first writer experiment uses canonical CMRC prompt images,
not this historical corpus. Corpus availability is not a historical-reading or
generation result. Historical forms, handwritten variation and etymological
language remain part of the broader model objective.

Reproduce the inventory on CPU with the existing source workspace:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_historic_glyph_inventory.py
```

Use `--db`, `--asset-root` and a new `--output` directory for another layout.
The full local manifest records path existence and sampled file hashes; this
public summary does not redistribute the original asset collection.
