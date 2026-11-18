# Modern passage exposure and Mandarin order preference

A writable training stream now exposes 2,127 unique modern Chinese passages as 1,049,411 successive glyph images. The original reader used these passages inside its question prompts. They supply 1,047,284 direct next-image predictions when each passage is read once. All passage slices match the original training pixels byte for byte. The index stores offsets and hashes, so the physical corpus stays in one place.

Two local transition models test the effect of the writing source on a further 6,062 ZhoBLiMP pairs. Each model learns from exactly 1,047,284 successor images. The test excludes the 552 previously scored physical pairs and their duplicates. Both approaches share a font, bitmap alphabet, smoothing rule, scoring method and held-out sentence pairs. The benchmark constructions were already known.

| Training source | Correct | Ties | Incorrect | Mean accuracy across 23 constructions |
| --- | ---: | ---: | ---: | ---: |
| Historical book passages | 2,660 | 1,609 | 1,793 | 57.6% |
| Modern QA passages | 3,450 | 329 | 2,283 | 59.5% |

The modern passages resolve more ties in both directions. Their advantage is 1.9 percentage points, with a descriptive 95% paired paradigm interval of −9.0 to +12.7 points. The modern accuracy interval is 49.7–69.1%. Both intervals miss the prespecified lower-bound criteria. This experiment leaves corpus replacement alone unqualified as a remedy for the neural reader. It establishes a reusable source of direct writing supervision and keeps autonomous language quality open.

The final CPU comparison took 7.71 seconds end to end, including source indexing and scoring. Fitting the transition counts took 0.45 seconds for books and 0.44 seconds for modern passages. The original neural evaluation continued on the GPU. The first CPU process exited with signal 15 before producing a case file. Its sender is unknown. The completed run uses a single exact multi-pattern corpus scan, checked against literal substring membership on 100 randomized fixtures. All pair counts and both macro accuracies were independently recounted. [Protocol](visual_passage_register_protocol.md) · [Measurements](evidence/visual_passage_register_20261118.json)

## Use the original passage pixels for training

Run `PYTHONPATH=. python scripts/prepare_visual_passages.py` on an installation containing the original CMRC sources, Noto training font and packed visual corpus. The index is saved to `data/visual_passages_v1/manifest.json`. After preparation, the loader needs only the index and the packed training corpus.

```python
from ilm.visual_lm.visual_passage_corpus import VisualPassageCorpus

passages = VisualPassageCorpus(
    'data/visual_passages_v1/manifest.json', 'data/visual_sequence_corpus_v1'
)
example = passages[0]
terms = model.training_terms(example['ink'].to(device), example['prompt_glyphs'])
loss = terms['code_nll'] + terms['stop_bce']
```

Each image tensor has shape (length, 1, 32, 32). The loader verifies the selected span's packed bits. The neural objective uses the first cell as a causal anchor.
