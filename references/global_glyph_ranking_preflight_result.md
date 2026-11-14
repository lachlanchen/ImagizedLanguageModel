# Whole-glyph ranking passes the real-source gradient and memory preflight

The new loss completes all four registered full-body cases. It uses the original
fresh joint writer and point-final continuation initialization, not a small
replacement network or a selected intermediate checkpoint. Both complete
five-glyph QA layouts match independent unsplit autograd across all 131 actual
learned tensors. The largest relative-L2 difference is 2.8995e-6; the largest
absolute difference is 4.1723e-7. Complete loss differences are below 7e-7.

| Complete case | Target glyphs | Image microbatches | Actual gradient tensors | Actual optimizer updates |
|---|---:|---:|---:|---:|
| QA, fixed | 5 | 3 | 131, matched to unsplit autograd | 0 |
| QA, proportional | 5 | 3 | 131, matched to unsplit autograd | 0 |
| Book, fixed | 513 | 9 | 131, finite and nonzero | 1, reset |
| Book, proportional | 513 | 9 | 131, finite and nonzero | 1, reset |

Each book update changes every learned tensor and creates all 131 actual Adam
states at counter one. All updates are reset. Frozen native parameters,
feedback/stop parameters and the four affine buffers remain unchanged under the
bound execution checks. This preflight exercises the new source-versus-blank
ranking term; the previous likelihood-only preflight remains separate and
terminal. The independent CPU unit cases also include mixed/all-white targets
and zero ranking strength; the four real-source cases happen to have nonempty
target tiles throughout and do not establish whitespace generalization.

The worker reports 98.195 seconds, no pressure waiting, 17.795 GiB peak allocated
VRAM and 59.294 GiB minimum available host RAM. It exits normally after the
four cases. The complete deployed model remains 7,553,448,370 parameters,
including 7,522,463,984 inherited parameters. Its 5,005,760 trainables do not
establish a compact foundation or an efficiency advantage.

This proves a computational precondition for the
[proposed objective comparison](global_glyph_ranking_decision.md). It does not
prove that ranking removes blank modes, improves visible writing, learns
language, stops naturally or generalizes to historical forms. No full ranking
fit or autonomous image-answer generation has run. A separate complete-corpus
source/input/resource binding is still required before that fit starts.

The [artifact export](evidence/global_glyph_ranking_preflight_result_20261114.json)
binds all reports, case files, initialization and sources. CPU validation
authenticates those artifacts and checks the reported numerical conditions; it
does not independently replay the neural computation or turn it into a language
score. Reproduce the export:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/summarize_global_glyph_ranking_preflight.py --verify
```

Keep the entire ILM goal active. The manuscript and native figures still need
the completed context-gradient failure, blank-mode measurement and this limited
preflight result, with their distinct evidence scopes preserved.
