# Read a marked pair of participant words

This 202,001-parameter English reader receives a verb span and two candidate spans in a 32-pixel glyph grid. It selects which candidate corresponds to the active subject and returns the original pixels for both words. Use zero-based, end-exclusive occupied cell offsets.

```sh
PYTHONPATH=. python -m scripts.read_natural_roles --source clause.png --cells 24 \
  --verb 9 15 --first 0 4 --second 19 23 --compact-paper \
  --actor actor.png --undergoer undergoer.png
```

The optional paper compaction merges consecutive unmarked blank cells. All output pixels come from the original coordinates. With both candidates supplied the reader selected correctly on 1,196/1,196 official test clauses. The separate open locator finds the two heads from the pictured sentence. [Open locator](../visual_role_locator/README.md) · [Tests and limitations](../../references/natural_pictured_roles_result.md) · [Source attributions](../../publication/ilm-nature/evidence/natural_roles_attribution.md)
