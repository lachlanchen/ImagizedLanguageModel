# A bilingual reader that answers from glyph images

Give the model a question PNG and a passage PNG. It writes a third PNG. The released optical network runs locally on CPU or CUDA and has 2,442,424 inference parameters including its book-trained backbone. Each image uses row-major 32 × 32 monochrome cells. The writer can reuse a sequence of input marks, select successive glyphs from its learned visual repertoire, and stop. Unfamiliar input shapes can be copied. Drawing previously unseen shapes from a description is outside the demonstrated capability.

## Read a pictured page

Run this example from the repository root. The two PNGs contain 11 question cells and 343 passage cells. The counts exclude padding in the final row. The reader writes the fish classification **辐鳍鱼纲鲈形目隆头鱼亚目雀鲷科** and stops after 15 cells.

```bash
example=models/paired_evidence/examples/cmrc/both/1
PYTHONPATH=. python -m scripts.write_paired_evidence \
  --question-image "$example/query.png" --question-cells 11 \
  --passage-image "$example/passage.png" --passage-cells 343 \
  --output /tmp/ilm-fish-answer
```

Open `answer.png` in the new output directory. The packed pixels in `answer.bin` have SHA-256 `0663b0989792b2a797e2f3ae9ecf0f679c333d13f2b86bbe66a0c5a803734786`. The optional JSON transcript uses known character labels for display. An unrecognized mark remains present in the physical image. For ordinary text, pass `--question` and `--passage` to render both with the preparation font. For open writing, provide the question or prefix alone. The default limit is 96 output glyphs. Use a fresh output directory.

The reader requires the released optical discourse weights and glyph palette already in this repository. Raw books, a remote model, a GPU and the training corpus are unnecessary for PNG inference. The page format currently assumes known cell boundaries and reading order. [Optical backbone](../visual_discourse) · [Archived successes and misses](examples)

## Train and evaluate

The fitting corpus combines four human QA sources with replay from the bilingual book and instruction writer. Prepare the underlying discourse corpus and the cursor preparation archive first. The source and archive checksums are recorded with the experiment. The three arms share 4,800 update draws. Existing verified fits are reused.

```bash
PYTHONPATH=. python -m scripts.prepare_paired_evidence
PYTHONPATH=. python -m scripts.train_paired_evidence
PYTHONPATH=. python -m scripts.train_paired_evidence_bytes
PYTHONPATH=. python -m scripts.evaluate_paired_evidence
PYTHONPATH=. python -m scripts.audit_paired_evidence
PYTHONPATH=. python -m scripts.replay_paired_evidence
```

The last command checks 16 real image replays on CPU and evaluates the exploratory historical-form transfer set, using the local source-plate archive. On 256 unseen modern passages the optical reader answered both human questions on 44. It correctly answered only two of 32 English dialogue questions whose four recorded wordings were all absent from the page. Open writing and historical etymology are unreliable. [Study and controls](../../references/paired_evidence_result.md) · [Corpus manifest](../../references/evidence/paired_evidence_v1/corpus.json) · [Backbone setup](../visual_discourse/README.md)
