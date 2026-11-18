# Chinese question answering by selecting visible passage glyphs

A CPU-trained reader recovered the exact complete answer image to **269 of 1,788 questions (15.0%)** about 425 held-out Chinese passages. It reads two streams of binary glyphs, estimates where the answer starts and ends, and emits that stretch of the supplied passage. Every returned pixel therefore has an identifiable source. Answer selection can still be wrong.

| Input to the trained reader | Complete exact answer images |
| --- | ---: |
| Original question, all held-out passages | 269 / 1,788 (15.0%) |
| Question replaced with equally many white cells | 18 / 1,788 (1.0%) |
| Original question on the exchangeable subset | 269 / 1,783 (15.1%) |
| Another question about the same passage | 11 / 1,783 (0.6%) |

The normal-minus-blank advantage was 14.0 percentage points (whole-passage 95% bootstrap interval 12.2–15.9). The normal-minus-exchanged advantage was 14.5 points (12.7–16.3). The registered extraction criterion passes. The reader still missed 1,519 complete answers. The comparison uses one fixed rendering and one division of the original training corpus. These passages were withheld from the extraction reader; the earlier neural generator had seen them in its training corpus. The generator and extractor therefore do not have comparable scores here.

Two boundary classifiers learned from 7,235 questions about 1,702 passages. Inputs include exact raster matches for runs of up to four adjacent glyphs, nearby question overlap and pooled ink density. The output is a passage interval with a learned length prior. The reader receives no character IDs, font, answer length or teacher states during deployment. Feature construction and fitting took 63.95 seconds. The complete run took 112.33 seconds with two CPU threads, 2.112 GiB peak RSS and a 108,624-byte compressed model. Corpus preparation and the prior supervised-task design are outside that timing. The first preflight stopped at the process RSS admission check; a fresh process completed both train-only fits, verified serialization and discarded those trial weights. The full fit ran once. An independent comparison checked all 5,359 predicted spans against the original packed cells. The measured advantage concerns extractive reading. Free language generation, cross-font transfer and matched-quality LLM efficiency remain open. [Protocol](visual_extractive_reading_protocol.md) · [Fit and controlled evaluation](evidence/visual_extractive_reading_20261118/result.json) · [Pixel audit](evidence/visual_extractive_reading_20261118/audit.json)

### What did correct training supervision add?

A subsequent matched fit reassigned answers among questions about the same training passage. It retained all 7,235 questions, all passage pixels and every passage's answer-length distribution. The reassignment changed 7,206 training answer images. On the original 1,788 held-out questions this reader got 59 exact answers (3.3%), compared with 269 (15.0%) for the correctly paired fit. Both readers answered 40 questions correctly, 229 were correct only with matched supervision and 19 only with rotated supervision. The paired advantage was 11.7 percentage points, with a whole-passage 95% interval of 10.1–13.5 points. This single reassignment control supports a contribution from correct question–answer pairing beyond the shared features and passage cues. [Control design](visual_extraction_supervision_control.md) · [Measurements](evidence/visual_extractive_reading_20261118/supervision.json)

### Two complete answer images

The first exact case in the fixed test order asks when 宏霸數碼集團 was founded. The returned image reads **1999年**. A separate process given just the trained reader and the two PNG inputs reproduced the reference PNG byte for byte. [Passage](evidence/visual_extractive_reading_20261118/first_exact_passage.png) · [Question](evidence/visual_extractive_reading_20261118/first_exact_question.png) · [Output](evidence/visual_extractive_reading_20261118/first_exact_answer.png)

The first case in the same order asks for the author of ChewingOSX / 酷音輸入法. The reader selected **作者** (“author”). The source answer is **高嘉良**. Both are legible, and only the name answers the question. [Passage](evidence/visual_extractive_reading_20261118/first_case_passage.png) · [Question](evidence/visual_extractive_reading_20261118/first_case_question.png) · [Output](evidence/visual_extractive_reading_20261118/first_case_answer.png) · [Reference](evidence/visual_extractive_reading_20261118/first_case_reference.png)

### Use the trained reader

From the project root with the project Python environment, run the following command. The counts exclude trailing padding in the supplied 32-column PNGs. An unpadded single-line PNG needs no count.

```bash
examples=references/evidence/visual_extractive_reading_20261118
PYTHONPATH=. python scripts/read_visual_passage.py \
  --model artifacts/visual_extractive_reader_v1/reader.joblib \
  --passage "$examples/first_exact_passage.png" --passage-cells 304 \
  --question "$examples/first_exact_question.png" --question-cells 13 \
  --output answer.png
```

A fresh checkout with the original project data can reproduce the fit with `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/train_visual_extractive_reader.py`. It writes the model, per-case span offsets and hashes, sample PNGs, and measurements in `artifacts/visual_extractive_reader_v1/`.
