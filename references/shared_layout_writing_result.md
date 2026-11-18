# Drawing a legible glyph and choosing the next word

One shared 4-by-4 ink layout increased the number of freely drawn images exactly matching a nonempty glyph from **198 to 649 of 668**. It reduced blank outputs from **348 to one**. The number of exact next characters remained **18**. This comparison used the same paired contexts from 334 Chinese development articles as the earlier experiment. The writer drew 也 correctly after 同时 in one passage. After 四川省历史文 it drew 线, where the article continued with 化. Asked about the evolution of 言, it wrote a lowercase a followed by a repeated oval shaped like a zero. [Complete output](evidence/shared_layout_writing_20261118/ordered_yan.png) · [Protocol](shared_layout_writing_protocol.md) · [Measurements](evidence/shared_layout_writing_20261118/result.json)

| Free successor across 668 contexts | Earlier patch writer | Shared layout | Shared layout, rotated training |
| --- | ---: | ---: | ---: |
| Matches a known nonempty bitmap | 198 | 649 | 668 |
| Matches a Han ideograph found in the Chinese source | 119 | 340 | — |
| Blank | 348 | 1 | 0 |
| Matches the recorded next character | 18 | 18 | 0 |
| Both next characters correct in the same article | 0/334 | 0/334 | 0/334 |

The ordered writer retained nine earlier exact responses, gained nine and lost nine. The paired article interval for the change in exact rate was −1.20 to +1.20 percentage points. A known bitmap can be a digit, punctuation or a letter. Under rotated supervision, every output matched a known nonempty bitmap and zero matched the recorded successor. Good physical form alone is a weak test of what the model read.

## Two constraints on language

The generator first selects a coarse arrangement of ink and then draws the fine patches in that arrangement. The training corpus contained 5,276 distinct successor bitmaps but only 539 layouts. Just 270 layouts belonged to a single training bitmap. Giving the correct source layout to the writer recovered 242 of the 668 recorded characters. A separate exact enumeration generated the greedy image for every one of the 539 selectable layouts. Together they could reproduce the source at the same 242 positions. Even a planner with perfect knowledge of the answer could not produce the remaining 426 through this trained greedy writer. The model's stochastic patch sampler has broader mathematical support. The enumeration applies to greedy generation. [Complete range audit](evidence/shared_layout_writing_20261118/support.json)

The context also has a hard limit. Its state retains the coarse signatures of the last two glyphs. When two prompts end in the same signatures, their next-image and stopping laws coincide exactly. After identical greedy outputs the states stay identical. Consequently this architecture cannot answer questions that differ earlier and end in the same two glyphs when those questions require different answers. More training under this representation cannot restore the missing information. The three article completions and the 言 response all reached 128 cells and repeated.

A candidate for general writing needs two information paths. A causal network with persistent access to the full pictured prompt predicts a distribution over successive glyphs. A coarse-to-fine writer receives both a shared layout and enough conditional fine detail to distinguish characters sharing that layout. An invertible rearrangement of pixels into detail patches supplies an exact reconstruction control. A learned writer must then reconstruct held-out pictured glyphs when supplied with the true detail, preserve question identity after a shared suffix, and produce complete answers when it must predict that detail itself. Spatial coherence, semantic selection and stopping have separate measurable failure modes.

## Training and inference

Both count models trained independently on 1,047,284 next-image observations from 2,127 passages. The ordered model occupied 2.08 MB compressed. Its fit after preparation took 0.21 seconds. The complete two-arm run, including preparation, checkpoint round trips, free rolls and source-layout conditioning, took 13.97 seconds and peaked at 0.75 GiB of process memory. A first process exited at its RSS check before opening the output directory. A fresh CPU process performed the full run once. These timings do not measure the cost to reach general-language accuracy. [Earlier model](patch_count_writing_result.md)

With the prepared corpus, train in a fresh experiment directory and use the standard text or aligned-image interface. Inference feeds back the actual emitted pixels; the input renderer is used only for typed prompts.

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. python scripts/train_shared_layout_language.py
PYTHONPATH=. python scripts/infer_patch_count_language.py \
  --model artifacts/shared_layout_language_v1/ordered/model.npz \
  --text '請說明「言」字的演變。' --output artifacts/my_layout_answer
```
