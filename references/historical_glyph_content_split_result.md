# Historical glyph partitions by source and image content

The earlier historical codec partitioned data by modern character. Distinct
modern characters can refer to the same historical glyph, and different SVGs
can produce identical 32×32 inputs. The same raster can therefore appear in
training and evaluation under different character labels. The audit reuses
the original cache after verifying its database hash, ordered SVG manifest,
record order and binary pixel geometry.

| Earlier V34 partition | Records | Records matching a nonblank training raster | Evaluated records | Evaluated records matching training |
| --- | ---: | ---: | ---: | ---: |
| Development | 4,789 | 524 | 4,096 | 500 |
| Sealed | 4,726 | 361 | 4,096 | 314 |

The evaluated counts follow the original evaluator's unshuffled row order
and recorded sample counts. The earlier reconstruction scores describe the
original character split, which includes these shared rasters. Reconstruction
accuracy after removing them remains unmeasured. See the [V34 result](continuous_glyph_codec_v34_result.md).

The new split forms connected groups from shared query characters,
traditional and simplified forms, source entry IDs, identical SVGs and
identical nonblank rasters. It hashes the sorted characters of each group
to an approximately 80% training, 10% development and 10% sealed allocation.
Five blank rasters are recorded separately. The resulting manifests point
to the existing raster rows and retain the corresponding source descriptions.
See the [entry recovery](historic_entry_metadata_result.md).

| New partition | Groups | Source entries | Nonblank glyph records | Entries with pronunciation, meaning and glyphs |
| --- | ---: | ---: | ---: | ---: |
| Training | 5,898 | 6,835 | 56,442 | 4,421 |
| Development | 756 | 893 | 5,460 | 576 |
| Sealed | 750 | 866 | 5,966 | 553 |

The independent check read all three manifests and verified all 67,868 raster
references against the cached pixels. It found zero shared exact rasters,
SVG digests, source labels, source entry IDs, linked modern forms or character
groups across partitions. Oracle, bronze, seal and Liushutong forms occur in
each partition. The base manifest records 877 additional glyph links that
were subsequently recovered from the HTML cache. See the
[partition evidence](evidence/historical_glyph_content_split_20261111.json)
and [raster addendum](historical_glyph_raster_addendum_result.md).

This partition controls exact duplication for a subsequent fit. Near-duplicate
shapes require further study. Prior checkpoints trained with other historical
partitions can have exposure to these held images, and foundation pretraining
exposure remains unknown. The present experiment measured partition overlap
and data availability. Historical reading and autonomous generation still need
separate model evaluations.

The manifests and verification receipt are in
`artifacts/historical_glyph_splits_v1/`. Reproduce them with the recovered
entries, original raster cache and V34 reports available.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/prepare_historical_glyph_splits.py \
  --output artifacts/historical_glyph_splits_reproduction
```
