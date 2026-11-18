# Predicting Chinese successor glyphs by drawing ink patches

A model trained on ordered Chinese passages drew the exact next character image in **18 of 668 contexts** from 334 previously unused articles. Rotating the future images within each training passage gave **0 of 668**. Each test paired two passages ending in the same visible glyph. Neither recorded successor appeared in either prompt. The ordered advantage was 2.69 percentage points, with a paired article bootstrap interval of 1.50–3.89. The 18 matches covered 14 distinct characters. This establishes a limited effect of local sequence supervision on real, freely generated strokes. [Protocol](patch_count_writing_protocol.md) · [Full measurements](evidence/patch_count_writing_20261118/result.json) · [Source and bitmap recount](evidence/patch_count_writing_20261118/audit.json)

The first exact prediction completed 各式不同 with 的. The first blank occurred after 年进行的, where the source continued with 调. The test counts only the specific image occurring in the source. Other sensible continuations are possible. Both models had seen the target shapes during preparation of the original physical repertoire. The test supports generalization to fresh contexts in one font. It leaves transfer to unfamiliar shapes and writing styles open.

| Outcome across 668 next images | Ordered training | Rotated training |
| --- | ---: | ---: |
| Exact recorded successor | 18 | 0 |
| Both independent successors correct in an article | 0/334 | 0/334 |
| Nonempty drawings | 320 | 93 |
| Drawings equal to a known nonempty glyph | 198 | 36 |
| Blank drawings | 348 | 575 |
| Distinct drawn bitmaps, including blank | 148 | 37 |
| Mean source ink F1 | .1365 | .0164 |

The writer divides each glyph into sixteen 8-by-8 ink patches. The input records which regions of each of the previous two actual glyph images have above-average ink. For each next patch, it combines counts from the preceding glyphs with counts of patches joining its already drawn upper and left neighbours. It learns small discrete visual parts and coarse input signatures. The saved model contains patch counts, patch values and stop counts. Answer generation requires neither a complete-glyph output table nor a font.

The ordered fit took 0.75 seconds after source preparation and produced an 18.98 MB compressed checkpoint. The control took 1.85 seconds. The complete two-arm run, including preparation, preflight, serialization, reload and evaluation, took 23.14 seconds with 0.86 GiB peak process memory. Each arm saw 1,047,284 direct successors and 2,127 genuine endings. These CPU timings do not establish Qwen parity or the proposed 1,000-fold saving.

## Independent writing

All eight free continuations reached the 128-cell cap. The ordered model produced recognizable fragments followed by repeated marks. The rotated model emitted only blank cells on all four prompts. Neither wrote an explanation about 言. A single model reader inspected the complete images with their conditions visible. The typed and pictured versions of the 言 question reproduced every output pixel from the ordered checkpoint in separate command-line executions. [Ordered answer](evidence/patch_count_writing_20261118/ordered_yan.png)

The trained local checkpoint is `artifacts/patch_count_language_v1/ordered/model.npz`. Use a fresh output directory for each inference. The optional `--sample-seed` selects ancestral sampling. The reported runs used greedy decoding.

```sh
PYTHONPATH=. python scripts/infer_patch_count_language.py \
  --model artifacts/patch_count_language_v1/ordered/model.npz \
  --text '請說明「言」字的演變。' --output artifacts/my_patch_answer
```

Replace `--text` with `--image PATH.png` to read a row-major page of 32-by-32 black-on-white glyph cells. `--input-glyphs N` excludes trailing padding. With the prepared visual corpus installed, both fits and their evaluations are reproducible using `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. python scripts/train_patch_count_language.py` in a checkout with a fresh experiment output path.
