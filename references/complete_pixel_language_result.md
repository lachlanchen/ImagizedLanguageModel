# Complete image-answer learning fits training questions but fails transfer

The completed native pixel-language actor produces **32/32 pixel-exact complete
training answers from original prompt PNGs**. On reused development data it
produces **0/16 exact Chinese answers and 0/8 exact English answers**. The raw
development outputs contain broken glyphs and occasional fragments of unrelated
training responses. This experiment establishes a working complete-answer
fitting path, not useful language generalization.

| Independent final assessment | Cases | Exact complete images | Foreground F1 | Learned stops | Cap failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training, fresh original PNGs | 32 | 32 | 1.000000 | 32 | 0 |
| Chinese development, normal questions | 16 | 0 | 0.239785 | 16 | 0 |
| Chinese development, question erased | 16 | 0 | 0.172495 | 16 | 0 |
| English development, normal prompts | 8 | 0 | 0.034482 | 8 | 0 |
| English development, prompt erased | 8 | 0 | 0.048394 | 8 | 0 |

Both answers match in all 16 training passage pairs and none of the eight
development pairs. Chinese normal-minus-erased F1 is 0.067290; English is
−0.013913. These pixel-overlap differences do not establish that question
meaning is understood. Exact reference pixels are a strict physical metric;
no independent semantic accuracy is inferred from it. All 24 normal and 24
erased development streams, their owning prompts and complete alternative
references were actually viewed. The qualitative failures are broader than
minor font or alignment differences. Cases 44 and 45 visibly reuse parts of
unrelated training answers.

## What learned and what was tested

The deployed path reads prompt pixels through the full native pretrained
context, predicts a physical image block, reads that emitted image as its own
history, and makes a learned stopping decision. A 32-by-64 binary output block
is losslessly represented by 256 adjacent 2-by-4 pixel pieces. Each piece has
256 possible binary patterns; these are physical patterns, not character IDs.
A causal normalized density scores every piece. Its inverse packing operation
contains no font, glyph inventory or answer table. The image codec only encodes
observed history; its image decoder is disabled in the output path.

Training scores every complete answer block. The implemented objective averages
piece NLL within blocks and blocks within questions, then adds per-question
balanced continue/end softplus at equal weight. This weighting differs from
the unscaled full joint answer NLL. Correct preceding images are supplied
during training. Autonomous assessment receives only the prompt and its own
emitted pixels, with learned stopping and a uniform 128-block cap. Neither a
reference answer nor its length enters generation.

The [registered protocol](complete_pixel_language_protocol.md) fixes 32 natural
Chinese questions from 16 passages, two distinct questions per passage, and 107
complete answer blocks. The model stops optimization at the registered
4,000-update assessment because all 32 answers are exact. Development outcomes
do not select that checkpoint. The 16 Chinese development questions and eight
English category representatives were fixed before learning; they are reused
development cases, not a sealed test. English is untrained transfer in this run.

| Registered training assessment | Exact complete answers | Foreground F1 | Correct-history whole-cohort image NLL |
| --- | ---: | ---: | ---: |
| Initial state | 0/32 | 0.130254 | 9.883952 |
| Update 1,000 | 17/32 | 0.943528 | 0.006959 |
| Update 2,000 | 29/32 | 0.996086 | 0.001138 |
| Update 4,000 | 32/32 | 1.000000 | 8.659788e-9 |

The NLL column uses correct preceding images. Image exactness and F1 use fully
autonomous generation. Intermediate assessments reuse authenticated native
prompt KV. The final training and every development stream invoke the PNG
reader afresh. All 32 complete native prompt prefills are bit-exact after
learning, confirming that the replayed prompt contexts remained valid.

## Execution and independent checks

The run retains 4,000 updates, 16,000 example draws and 500 exposures per
question. An actual update-two restore/replay is bit-exact for model, Adam,
sampler, loss and gradient measurements. It adds one executed call and four
examples: the durable ledger contains 4,001 calls and 16,004 executed examples.
Recorded active optimizer time is 3,789.292 seconds (63.2 minutes). Total run
time, including initialization, prefill, checks and final assessment, is
4,893.048 seconds (81.6 minutes). Peak allocated GPU memory is **18.191 GiB**
on GPU0, an RTX 4090 D. GPU1 and its separate service were preserved.

The independent CPU audit validates every source/data binding, original prompt
and reference image, sampler permutation, exposure ledger, output tensor, PNG
and stopping event. It re-encodes **668 actual emitted-image feedback records**;
the maximum CPU-versus-recorded-GPU feature error is **6.914139e-6**, within the
registered 1e-4 tolerance. All 208 complete streams across initial,
intermediate, final and development assessments retain their original pixels.
This auditor does not independently replay the full inherited reader or
optimizer, and it does not assign semantic correctness.

A separate cold deployment process restores only this actor's own final
learned state plus authenticated frozen source weights, history encoder and
numeric controls. The first four original training PNGs and the first longest
complete training answer are selected without quality ranking: indices
0, 1, 2, 3 and 28. All five replays match complete pixels, stop probabilities,
history-feature hashes and reader-memory hashes exactly. Expected output
tensors are accessed only after generation is saved. No legacy training or
raster checkpoint, online teacher, runtime OCR, tokenizer or font answer
renderer enters this deployment path. The cold check peaks at 17.529 GiB,
takes 76.547 seconds to load and 96.675 seconds in total. The process exits
and releases GPU0. This verifies deployment reconstruction, not transfer.

The actor retains **7,543,984,691 total parameters**, with **13,855,681 trained**.
Its substantial external language/vision pretraining and prior raster learning
are excluded from the local timings. This is not a lightweight foundation
trained from scratch or a matched-quality efficiency comparison with an LLM.
Different initialization, decoder, precision and schedules also prevent
attributing improvement over the earlier writer to the likelihood alone.

## Evidence and next decision

The [independent audit](evidence/complete_pixel_language_result/independent_audit.json),
[producer report](evidence/complete_pixel_language_result/producer_report.json)
and [raw-image review manifest](evidence/complete_pixel_language_result/review_and_manifest.json)
retain all cases. The source-first training pair is visible in the
[first training plate](evidence/complete_pixel_language_result/final_cases_00_07.png).
The first development pair and controls are retained as
[case 32](evidence/complete_pixel_language_result/development_case_32.png) and
[case 33](evidence/complete_pixel_language_result/development_case_33.png).
Every other case remains alongside them, including all failures. The
[cold deployment receipt](evidence/complete_pixel_runtime_qualification/report.json)
binds its own final checkpoint and five actual fresh-PNG calls.

The final checkpoint is local at
`artifacts/complete_pixel_language_v1/execution/checkpoint_latest.pt`, SHA-256
`b68f531bbf56588167bf16f3cd1a133a5e35dbed69f208cb62a8549b900ff9f5`.
[Inference commands](complete_pixel_language_inference.md) use that completed
checkpoint. Arbitrary new typed/image prompts remain unsupported as a useful
language capability, even though the input interface accepts them.

Further unchanged fitting of the same 32 questions is not the next experiment:
training already reaches exact reproduction. The next learning stage must
test reusable glyph generation and conditional language selection on varied
examples, with new-prompt output quality as the criterion. More diverse data
is a testable candidate, not a demonstrated repair; earlier larger-corpus
writers already failed under different recipes. The general image-language
objective and the bounded manuscript/figure revision remain unfinished.
