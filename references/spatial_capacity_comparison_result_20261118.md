# Spatial capacity resolves the continuous ceiling, but binary fidelity fails

Both spatial codecs complete 6,000 updates on the unchanged physical training
partition. Increasing the features at each of sixteen sites from four to
sixteen raises continuous validation F1 from .929688379 to .996423850 and binary
F1 from .847344074 to .943690071. The common convolutional initialization,
balanced source order, objective and optimizer budget remain fixed. Each new
codec has 1,770,449 parameters, 3,084 more than the spatial-64 model.

| Physical measurement | Continuous | Binary |
|---|---:|---:|
| Training pooled F1 | .999996189 | .981661645 |
| Validation pooled F1 | .996423850 | .943690071 |
| Validation macro-form F1 | .993422244 | .928693592 |
| Validation exact cells / 4,907 | 2,745 | 94 |
| Validation exact forms / 952 | 316 | 3 |
| False blanks / false ink, complete source | 0 / 0 | 0 / 0 |

The binary arm fails four of ten fixed conditions: validation pooled F1,
validation macro F1, validation macro exactness and retention of the continuous
ceiling. Its training F1, continuous ceiling, .02 gain and absence of false
blanks or false ink pass. Thirty-two binary plan events plus actual XOR-tree
events average 49.475474 on the complete source, passing the unchanged total
event condition against the old direct tree's 97.302489. Validation alone costs
72.474424 events. These are event counts, not bytes, throughput or language cost.

The continuous result establishes faithful supplied-image reconstruction on
this reused physical development split. Its exact-form rate is only .331933;
high pooled F1 must not be called exact copying. The hard representation still
loses detail. Neither result establishes conditional selection, semantic
understanding, autonomous writing or whole-model efficiency. No conditional
language fit is admitted, and the original language criteria remain unchanged.

Both complete-source gradient/update/reset probes and both update-three
optimizer replays pass. The [independent audit](../scripts/export_spatial_capacity_comparison.py)
recounts all 275,792 saved initial/final images and XOR/event arrays, all 12,000
retained updates, checkpoint hashes, initial shared convolution values and
source-order identity. Old controls are not rerun. The [complete result](evidence/spatial_capacity_comparison_result_20261118.json)
and fixed [validation](evidence/spatial_capacity_comparison_20261118/held_fixed_reconstructions.png)
and [training](evidence/spatial_capacity_comparison_20261118/train_fixed_reconstructions.png)
plates retain actual pixels without fonts, OCR, repairs or case selection.

The neural run takes 256.635 monitored seconds, .524 GiB peak allocation and
48.839 GiB minimum sampled available host memory. Supervisor time is 265.266
seconds. The separate CPU recount takes 29.309 seconds at 1.705 GiB peak RSS.
All owned model processes terminate normally. No contextual actor is loaded.

This binary recipe is terminal. Preserve the substantial physical improvement
and the failed absolute criteria together. A later equal-bit quantization
comparison may test scalar precision while retaining spatial geometry, source
allocation, losses, exposure, actual event cost and every original fidelity
condition. More unchanged epochs or a chosen intermediate checkpoint are not
an intervention.
