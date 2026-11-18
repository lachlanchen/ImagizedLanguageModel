# Writing two different answers from one Chinese passage

Asked about the musical style of Stam1na, the model wrote **前卫金属**. Asked when the band formed, it wrote **1996年**. The same 453-cell passage and both questions were shown in Noto Sans CJK. Each answer matched every pixel of a human reference rendered in Noto Serif CJK. This was the first successful pair in the predetermined evaluation order. [Pictured passage](evidence/grounded_detail_examples_20261118/pair_passage.png) · [Musical-style answer](evidence/grounded_detail_writing_20261118/pair_answer_a.png) · [Founding-year answer](evidence/grounded_detail_writing_20261118/pair_answer_b.png)

The system uses the previously trained full-question passage reader to select start and end positions. A new model writes each selected cell from a coarse ink layout and locally observed strokes. The answer typeface exists only in preparation and evaluation. Standalone inference from the pictured passage and musical-style question reproduced the evaluated output byte for byte. The answer can extend across several words. Its content and stopping position come from the selected contiguous span. Free writing and etymological synthesis remain open. [Protocol](grounded_detail_writing_protocol.md) · [Original training of the reader](visual_extractive_reading_result.md)

| Official CMRC development input | Eligible questions | Correct span in Sans | Complete exact Serif answer |
| --- | ---: | ---: | ---: |
| Intact question and passage | 3,170 | 565 | 124 |
| Another question about the same passage | 3,158 | 26 | 9 |
| Blank question | 3,170 | 31 | 11 |
| Blank passage | 3,170 | 0 | 0 |

The 3,170 questions came from 836 pictured passages. Of 3,219 original questions, 49 had unsupported input characters. The writer reproduced 124 of the 565 correctly selected spans in the new typeface. All 124 differed from a direct copy of the source span. Restricting the same writer to the coarse layout gave one complete match. The two complete answers were both correct in seven of 3,100 different-answer pairs. In the first failure, a question asked which two companies developed a game. The reader selected a longer span containing both names and unrelated text. [Failure image](evidence/grounded_detail_examples_20261118/failure_output.png) · [Reference](evidence/grounded_detail_examples_20261118/failure_reference.png)

The detailed writer gained 3.88 percentage points over the coarse writer (95% interval by passage 3.23 to 4.57). On the exchangeable subset, keeping the original question gained 3.61 points (2.94 to 4.31). Every accepted human wording was eligible to match. The evaluation used an official development set whose article text had previously been used in successor diagnostics. The independent DRCD pool remains reserved. A separate recount checked all 25,336 output streams in both writing conditions against the recorded images and confirmed all complete scores and seven pairs. [Measurements](evidence/grounded_detail_writing_20261118/qa.json) · [Pixel recount](evidence/grounded_detail_writing_20261118/recount.json)

## What the model learned about strokes

The writer saw 952,606 paired glyph occurrences drawn from 2,127 training passages. Of 5,276 distinct source bitmaps, three mapped to more than one target bitmap. On the 537 withheld nonempty pairs, taking the most common target patch for each observed source neighbourhood gave less stroke overlap than a direct copy. A convolutional network learned to correct the input strokes in patches missing their most specific training observation. The combined system preserved every drawing on familiar training pairs.

| Assay, mean ink F1 | Copy source | Local counts | Neural fallback with counts |
| --- | ---: | ---: | ---: |
| 4,742 familiar nonempty pairs | .7190 | .9844 | .9844 |
| 537 withheld pairs | .7239 | .6596 | .7510 |
| 314 additional Han shapes absent from the entire training text | .7334 | .6712 | .7565 |

On the 537 withheld pairs the gain over copying was .0271 (95% interval .0231 to .0312). It missed the registered minimum gain of .03. The system wrote zero exact reference bitmaps on the 537 withheld and 314 additional pairs. Rescoring the new images under all four question conditions gave the same 124 intact complete answers and seven both-correct pairs. The original count writer remains the qualified grounded component. [Neural protocol](unseen_detail_residual_protocol.md) · [Physical measurements](evidence/grounded_detail_writing_20261118/residual.json) · [Complete answer comparison](evidence/grounded_detail_writing_20261118/residual_qa.json)

## Running the model

The independent local deployment uses a 108,624-byte trained reader and a 2,630,936-byte count writer. The earlier reader fit took 63.95 seconds after corpus preparation. The new count fit took 0.21 seconds; its preparation, fitting and physical evaluation together took 12.52 seconds. The complete primary question evaluation took 125.19 seconds. The optional 10,833-parameter neural correction took another 92.65 seconds to fit. All timings are on CPU; none establishes a speed ratio to an LLM at matched independent language quality.

From a checkout with the prepared corpus and fitted reader, use fresh experiment output directories for reproduction. The following commands fit the count writer, score its answers and recount the original outputs.

```sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
python scripts/fit_grounded_detail_writer.py
python scripts/evaluate_grounded_detail_writer.py
python scripts/recount_grounded_answers.py
```

The saved pictures and the trained local weights can be used directly for inference.

```sh
PYTHONPATH=. python scripts/infer_grounded_detail_writer.py \
  --reader artifacts/visual_extractive_reader_v1/reader.joblib \
  --writer artifacts/grounded_detail_writing_v1/writer.npz \
  --passage-image references/evidence/grounded_detail_examples_20261118/pair_passage.png \
  --passage-cells 453 \
  --question-image references/evidence/grounded_detail_writing_20261118/pair_question_a.png \
  --question-cells 18 --output artifacts/my_grounded_answer
```

Use the optional text input flags to rasterize a typed passage or question. To reproduce the transfer experiment after the count trial, run `scripts/fit_unseen_detail_residual.py`, followed by `scripts/recount_grounded_answers.py --residual` with the same environment. Pass `--residual artifacts/grounded_stroke_residual_v1/neural.pt` to inference to use the measured fallback.
