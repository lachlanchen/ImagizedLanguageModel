# Visual span writing on held books and word origins

The visual-span system translated **24/192** pictured sentences with complete meaning according to two masked source-only readings. The independent 10.48M-parameter single-glyph writer earned **8/192**, a matched image exchange **0/192**, and quantized Qwen3-8B supplied with text **162/192**. The gain over the single-glyph writer was 8.33 percentage points (four-book bootstrap 3.13–15.63). The new model inherits a translation decoder, lexical projection weights and training-time subword boundaries. The result measures what can be transferred through a visual interface. Independent visual pretraining, broad language understanding and a compute advantage at equal quality are outstanding. [Article](../publication/ilm-nature/manuscript.pdf) · [Original inputs and 576 book answers](../publication/ilm-nature/evidence/span_rollout_answers.zip) · [Votes and matched comparisons](evidence/optical_span_rollout_v1/assessment.json)

| Primary assessment | To English / 96 | To Chinese / 96 | Both / 192 |
|---|---:|---:|---:|
| Span writer | 16 | 8 | 24 |
| Single-glyph writer | 2 | 6 | 8 |
| Span writer, exchanged source | 0 | 0 | 0 |
| Qwen3-8B Q4, text | 86 | 76 | 162 |
| Corpus reference | 52 | 64 | 116 |

Four project-held works supplied 24 paired passages each. Sentimental Education, Walking, Stray Birds and The Sound and the Fury contributed 7, 3, 2 and 12 credited span translations respectively. The paired interval over exchanged input was 5.21–20.31 percentage points. With references supplied in a separate assessment, counts were 13 for the span writer, three for the single-glyph writer, 103 for Qwen and 191 for the references. External pretraining overlap is unknown. Scoring was automated. All 576 book answers reproduced from their original input pixels on CPU, including boundaries and stop decisions. [Reservation](evidence/optical_span_rollout_v1/panel.json) · [CPU replay](evidence/optical_span_rollout_v1/replay.json)

## What the model learned

The fitted conditional frequencies use 338,663 English-target and 324,045 Chinese-target occurrences from 28,342 bilingual training pairs. A projected contextual state assigns mass to the spans attested for each lexical alternative, stratified by initial or later position. The writer chooses a span without supplied boundaries, draws its cells with the compiled neural motor, then uses the feedback prototype obtained from the actual emitted bitmap identities. The model supplies its own stop. Per direction the retained contextual decoder has 25.49M parameters, the optical source reader 3.22M parameters and the projection and feedback tables another 11.52–12.92M floating values. The motor before compilation has 2.27M parameters. Full upstream checkpoints each have 77.94M parameters at load. The book experiment ran on one RTX 4090 D. Its 14.2-second generation time includes setup, three conditions and PNG archival. It excludes antecedent training and the comparator/judge. [Deployment source](../ilm/visual_lm/optical_span_rollout.py) · [Fits and costs](evidence/optical_span_rollout_v1/generation.json)

The finite-mass surface-regret identity in Lean establishes the one-step advantage of choosing the most probable optical surface under the estimated distribution. On all 6,388 development decisions with supplied histories, the maximizer was also the modal spelling of the single best retained lexical alternative. The one-step pooling advantage was zero on these observations. The visible quality gain over the independent writer changes several things together, including inherited weights. [Lean proof](../formal/span_list/SurfaceDecision.lean) · [All measured decisions](evidence/optical_span_rollout_v1/surface-mass.json)

In a separate masked assessment using one contextual model per direction, the native text interfaces scored 81/192, replacing its source with the optical reader gave 31/192, and replacing its output with the span writer gave 24/192. The book intervals for the losses were 18.23–33.85 points at the source and 0–7.81 at the output. A subsequent development-only optical source-path fit recovered the entire native encoder address sequence on 196/240 Chinese inputs and 198/240 English inputs, versus 167/240 in both directions for the longest available edge. This source fit qualifies for a fresh end-to-end test. It uses training-time text addresses and has yet to write the book answers in this report. [Interface votes](evidence/optical_span_rollout_v1/interface-votes.json) · [Source-path fit](evidence/optical_span_rollout_v1/source-paths.json)

## Etymology and controlled translation

Eight contrast pairs across the two directions tested roles, polarity, quantity and temporal order. The span writer earned 7/16 individual credits and 2/8 complete directional contrasts. The single-glyph writer earned 0/16, and the reference translations 14/16. Neither writer answered the six English-origin or eight Chinese-origin questions correctly on inspection. Eight further pictured requests asked for attested forms in chronological order. Neither writer drew a correct sequence. The 12 distinct requested specimens were absent from the corresponding output repertoires. Extending physical support by copying from the source is possible, but selecting the right source and explaining it remain separate learning problems. [Actual 76 additional answers](../publication/ilm-nature/evidence/word_origin_probes.zip) · [Scores and review method](evidence/optical_span_rollout_v1/word-origin-scores.json) · [CPU replay](evidence/optical_span_rollout_v1/word-origin-replay.json) · [Support theorem](../formal/span_list/VisibleWriterSupport.lean)

The Chinese source records were 言 (E07105), 木 (E04338), 山 (E05416) and 日 (E05273) in the project's Hanziyuan collection. English reference facts were checked against the etymology entries for [window](https://www.merriam-webster.com/dictionary/window), [alphabet](https://www.merriam-webster.com/dictionary/alphabet) and [school](https://www.merriam-webster.com/dictionary/school). A separate older independent model previously selected labelled historical specimens but produced a single mark when asked for an explanation. [Earlier historical selection study](visual_story_writer_result.md)

## GPU inference cost at fixed output

Both directional writers ran on one RTX 4090 D. Each traversal translated all 192 original requests, grouped by direction. The largest batch held all 96 requests for one direction. All 2,304 answers across the twelve timed passes reproduced the archived bitmaps, boundaries and stops.

| Batch size | Median requests/s | Median board J/request |
|---:|---:|---:|
| 1 | 6.51 | 14.810 |
| 12 | 33.86 | 2.975 |
| 48 | 99.43 | 1.169 |
| 96 | 186.76 | 0.722 |

Batch 96 gave 28.7× the throughput of batch one and used 20.5× less board energy per request. These are within-model comparisons. The measurements cover warm inference, including transfers to and from the device. They exclude model preparation, PNG encoding, the host's electricity and external judging. The three batch-one traversals took 15.29 to 30.56 seconds, so the individual trials matter when interpreting the ratio. No matched-quality LLM efficiency factor has been established. [All trial times, energy and peak allocation](evidence/optical_span_rollout_v1/gpu-cost.json) · [Protocol](span_gpu_cost_protocol.md)

## Reproducing an answer

From the repository root, use the project Python environment with PyTorch, transformers, huggingface_hub, safetensors, Pillow and the repository's data tools. The loader uses locally cached Helsinki-NLP checkpoints. The English-to-Chinese revision is `408d9bc410a388e1d9aef112a2daba955b945255`; the Chinese-to-English revision is `cf109095479db38d6df799875e34039d4938aaa6`. Geometry is one to 96 occupied cells of 32 × 32 pixels in row-major order. The following example reads the original PNG and writes a new one.

```sh
python -m scripts.write_optical_span \
  --input publication/ilm-nature/figures/span_rollout/0003-source.png \
  --cells 27 --to zh --output answer.png --device cpu
cmp answer.png publication/ilm-nature/figures/span_rollout/0003-span.png
```

The published fit counts and previously released reader and motor checkpoints suffice for this example. To regenerate the study with the local bilingual library and comparator models, run the scripts in order: `reserve_optical_span_rollout`, `fit_optical_span_rollout`, `evaluate_optical_span_rollout`, then `evaluate_optical_span_rollout --replay` using `python -m scripts.<name>`. The separate interface and word-origin probes have their own frozen protocols and archives. [Protocol](optical_span_rollout_protocol.md) · [Interface controls](optical_span_control_protocol.md) · [Instruction probes](span_instruction_probe.md)
