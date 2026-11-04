# Grounded visual reading and image answers

## Complete observed image blocks

V58 adds a probability law over variable-length contiguous blocks of actual
prompt images, marginalizing equal sequences. Its predicted end position
terminates an answer. It gets 25/256 exact development answers (9.766%), versus
21/256 for matched additional glyph training; the exact-match difference is
unresolved. Raster F1 is 27.033%, and only 4/393 paired questions have both
answers correct. The reading qualification fails. See the
[V58 measured result](../references/visual_answer_blocks_v58_result.md).

The inference-only model runs locally from a native image. No initial training
checkpoint, text vocabulary, OCR or external model call is needed:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/infer_visual_answer_blocks.py \
  --checkpoint artifacts/visual_answer_blocks_v58_20261105/block/checkpoint_final.pt \
  --prompt-image prompt.png --out answer.png --export ilm_blocks_standalone.pt \
  --device cuda:0
```

The export is 78,772,107 bytes. Replaying the emitted `_prompt.png` with that
export reproduces the answer pixels and source-boundary events exactly. PNG
cells must follow the same 32-pixel alignment described below. Typed prompts
can use the host rasterizer through `--prompt-text`. The `.pt` output is a
dictionary with an `images` tensor. `--continuation` selects the retained
glyph copy/free writer, whose useful general language quality is unqualified.

For reproducible V58 training, run `train_visual_answer_blocks_v58.py` with
`--arm block` and then `--arm glyph` in separate output folders, and complete
both before `eval_visual_answer_blocks_v58.py`. Each uses 4,000 updates on
GPU 0. The current data hashes and initialization are checked automatically;
the production protocol rejects modified inputs or replaced final outputs.

## Frozen visual foundation comparison

V59 tests identical small learned block heads over frozen student or released
MIXAR features. It is not a deployment upgrade: the larger feature source gets
6/256 exact answers versus 24/256 student, under our shifted Noto CJK interface.
See the [negative result and resource accounting](../references/frozen_visual_foundation_v59_result.md).
Cached readout inference cost excludes the encoder. External pretraining must
not be counted as single-GPU training here. Shared release weights are reused.

## Per-glyph copy/free baseline

V57 trains a compact causal image reader with two output paths: copying a
currently visible image and generating a new binary raster. A learned stop
hazard controls answer length. The runtime receives image tensors and packing
lengths; it receives no strings, Unicode IDs, OCR transcript, vocabulary or
external model output.

The completed registered run gets 10/256 exact held-out answers, versus 1/256
with the question blanked and 0/256 for its matched no-copy control. Raster F1
is 15.718%; only 1/393 paired questions has both answers correct. All 256
supplied motor images still copy exactly. The reading gates fail. See the
[complete measured result](../references/visual_copy_language_v57_result.md).

This is an experiment in Chinese paragraph reading. It does not qualify
general instruction following or historical-script generation. The aligned
32-pixel-cell input also does not yet segment a raw book page. The copy path
preserves an arbitrary input form when selected; it does not by itself learn
that form's meaning.

## Data

Prepared data lives in `data/visual_qa/cmrc2018_short/`: `train.jsonl`,
`development.jsonl`, `counterfactual_pairs.jsonl`, `LICENCE` and
`preparation_report.json`. Raw pinned source files are in `data/raw/cmrc2018/`.
The subset has 3,211 training and 1,321 development questions, full paragraphs
with prompts up to 480 cells, and answers up to 16 cells. Article-title and
context overlap are removed across splits. Data is CC-BY-SA-4.0; cite
[Cui et al., EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1600/).

Rebuild the same subset with:

```bash
PYTHONPATH=. python scripts/prepare_visual_reading_qa.py
```

Host rendering is deterministic Noto Sans CJK, 26-pixel writing inside a
32-pixel cell. Images are binarized at 0.5. Question strings and annotated spans
remain in preparation/evaluation metadata. Only rendered images enter the
network. The primary loss marginalizes duplicate source images rather than
assigning a persistent character identity.

## Train and audit

The registered comparison starts from the V55 joint EMA. Both arms make
4,000 updates, batch four, with identical data choices. Three updates use
natural reading questions; the fourth retains natural Chinese continuation
and supplied-image motor exercises. Reader, density and output heads train
together. The no-copy control has the same foundation and budget with its
copy component disabled.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_visual_copy_language_v57.py \
  --arm copy --out artifacts/visual_copy_language_v57_20261105/copy --device cuda:0

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_visual_copy_language_v57.py \
  --arm no_copy --out artifacts/visual_copy_language_v57_20261105/no_copy --device cuda:0

# Run only after both arms finish.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_visual_copy_language_v57.py \
  --run artifacts/visual_copy_language_v57_20261105 --device cuda:0
```

Training uses deterministic CUDA algorithms and math SDPA. Each output folder
contains current/final checkpoints, optimizer and RNG state, source hashes,
training metrics and a report. Use `--resume path/to/checkpoint_latest.pt` with
the same arguments after interruption. Completed final checkpoints and audits
are protected against replacement. `--smoke` runs four updates and must never
be cited as language-quality evidence. Training and audit artifacts remain
ignored by Git; source, protocols and summarized evidence are committed.

The audit measures exact stopped answer sequences, multiset raster F1,
blanked-question controls, a matched no-copy model, paired questions sharing
the same paragraph, and preservation of supplied motor images. F1 alone does
not measure word order. The generation function receives neither the answer
nor its length. The fixed 17-cell safety cap is recorded separately from
learned stopping. See the [frozen protocol](../references/visual_copy_language_v57_protocol.md)
for thresholds, source hashes and limitations.

## Independent inference

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/infer_visual_copy_language.py \
  --checkpoint artifacts/visual_copy_language_v57_20261105/copy/checkpoint_final.pt \
  --prompt-image prompt.png --out answer.png --export ilm_reading_standalone.pt \
  --device cuda:0
```

The native PNG must contain an aligned grid of 32-pixel cells, without gutters
or unused trailing cells. Read order is left to right and then top to bottom.
The saved `_prompt.png` is a single row containing precisely the input cells,
so it can be replayed directly. `--prompt-text` provides optional host-side
rasterization, for example `资料：……问题：……回答：`; no text enters model code.

The primary decoder stops when its learned hazard reaches 0.5. Otherwise it
chooses the copy component when the learned gate reaches 0.5, sums attention
over identical observed forms and emits the highest-mass image. The free
component uses raw-pixel beam eight. This hierarchical choice is not global
maximization of the complete image mixture.

Outputs include the actual answer PNG, exact raw image tensor, input PNG and
a JSON receipt with copy/free routes and stopping probabilities. A stop-only
answer has zero raw cells and a white PNG placeholder explicitly marked in
the receipt. The export embeds the complete model; it does not require the
initial foundation checkpoint at inference.
