# Recovered later entries and narrow bronze glyphs

The older importer saved historical SVGs from the first etymology entry for
each query. Its HTML cache also contained 877 SVG links in later entries for
272 characters. The recovery authenticated each cached page and matched every
SVG by source entry, stage, label and digest. It rasterized all 877 SVGs at
32×32 with the same margins and threshold as the earlier codec. See the
[metadata recovery](historic_entry_metadata_result.md).

| Partition | Added entries | Added 32×32 rasters | Base and added rasters |
| --- | ---: | ---: | ---: |
| Training | 236 | 770 | 57,212 |
| Development | 35 | 58 | 5,518 |
| Sealed | 29 | 49 | 6,015 |

All additions retained their source family's partition. The independent
verification checked every SVG and raster digest and found zero exact SVG,
raster or source label overlap across the combined partitions. The recovered
vectors, binary raster tensor and three partition manifests are in
`artifacts/historical_glyph_addendum_v1/`. Each addition's `raster_index`
indexes `later_rasters.pt`. The original manifests index the original cache.
See the [content-grouped split](historical_glyph_content_split_result.md).

## Fine strokes in tall glyphs

In the base partition, 65,128 distinct SVGs produced 65,128 distinct
nonblank 32×32 rasters. Five other SVGs produced all-white rasters. All five
are tall bronze forms. With aspect ratio preserved and proportional margins,
the available width for their strokes becomes very small. A binary threshold
of 128 then removes all ink at the smallest resolution. The same vectors
yield visible ink at 64×64, where several strokes remain broken. The 128×128
images preserve more of the outlines. See the
[reviewed raster comparison](evidence/historical_raster_resolution_20261111.png).

| Source glyph | Ink pixels at 32×32 | At 64×64 | At 128×128 | Ink width at 128×128 |
| --- | ---: | ---: | ---: | ---: |
| B00160 | 0 | 22 | 131 | 19 |
| B00161 | 0 | 33 | 169 | 28 |
| B11887 | 0 | 57 | 308 | 23 |
| B12642 | 0 | 34 | 238 | 18 |
| B10991 | 0 | 7 | 204 | 25 |

For these five 128-pixel-high examples, a 32×128 crop retains every ink pixel.
Pasting each crop at its recorded horizontal offset into a white 128×128
canvas reproduces the original image exactly. A tiled image representation
can store each crop in four 32×32 tiles. The square canvas takes sixteen.
These are image representation measurements. Learning to generate writing
over several tiles and evaluating the resulting forms remain model tasks.
The compact images and offsets are in the addendum's verification receipt.
See the [recovery and resolution evidence](evidence/historical_glyph_raster_addendum_20261111.json).

To reproduce the SVG recovery and resolution board with the cached source
and base split available, select a new output directory.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/recover_historical_glyph_addendum.py \
  --output artifacts/historical_glyph_addendum_reproduction
```
