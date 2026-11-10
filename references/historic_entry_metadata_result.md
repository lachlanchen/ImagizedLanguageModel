# Historical glyphs paired with source descriptions

The legacy database has 9,055 character rows, all with empty pronunciation
and meaning fields. The Hanziyuan AJAX fragments store the query character
in an `etymology-nav` element and put the labelled fields inside individual
etymology entries. The earlier parser looked in the document title and large
headings for the character. It also stopped at the first historical glyph
section when a fragment contained several entries.

The parser now reads those labelled fields and resolves historical glyph
styles within each entry. The existing ingestion interface uses the first
entry in source order for its single character summary. The offline
recovery retains each entry independently, including traditional and
simplified forms, pronunciations, meanings, usage and decomposition when
the source provides those fields. It links a glyph only when the character,
stage, label and SHA-256 of its embedded SVG match the curated inventory.
See the [stage curation](historic_stage_curation_result.md).

| Recovered data | Count |
| --- | ---: |
| Cached AJAX fragments examined | 9,055 |
| Characters with source entries | 7,755 |
| Individual source entries | 8,594 |
| Characters with multiple entries | 696 |
| Curated glyph records linked by content | 67,873 |
| Entries with pronunciation, meaning and a matched glyph | 5,550 |
| Source glyph links without a matching saved asset | 877 |

All 877 unmatched links occur in later entries for 272 characters. The cache
contains their SVG data, and the manifest records their digests. For example,
the legacy assets for 只 cover 135 glyphs in
its first entry. The cached fragment supplies another 71 across its later
entries. The recovered entry for 言 links all 66 saved glyphs to its own
description. The audit found zero differing SVG digests among source links
with the same character, stage and glyph label. All 877 links were recovered
and rasterized in the [historical glyph addendum](historical_glyph_raster_addendum_result.md).

The recovery runs locally from the existing cache and uses a read-only
database connection. The complete entries and source receipts are in
`artifacts/historic_entry_metadata_v1/`. The public summary records the
counts and hashes. See the [recovery evidence](evidence/historic_entry_metadata_20261111.json).

The associations preserve the site's claims and images. Independent
verification of meanings and dates remains a separate research task. The
historical training split now keeps shared SVG content, exact rasters and
recorded variants together. The active CMRC writer experiment uses its
separately registered corpus. See the [partition audit](historical_glyph_content_split_result.md).

To reproduce the recovery with the legacy cache and curated inventory
available, choose a new output directory.

```bash
PYTHONPATH=. python scripts/recover_hanziyuan_entry_metadata.py \
  --output artifacts/historic_entry_metadata_reproduction
```
