# Training and running the exact raster language model

This implementation consumes writing images, predicts a normalized
distribution of future writing pixels and rereads its output images. Its native
interface has no tokenizer, OCR transcript, character inventory or output lookup.
Optional text input is rendered before entering the network.

The model is a research implementation with measured copying and context
sensitivity. It has not qualified as a useful conversational language model.
The [current evidence](../publication/ilm-nature/README.md) distinguishes that
status from the complete, executable training and inference path.

## Probability model

Each cell is a binary 32×32 ink image. A causal visual reader maps an ordered
history of cells to a continuous hidden state and image field. The field has
an exact inverse that supplies aligned pixel logits. A causal row transformer
and a masked within-row network add learned residual logits:

```text
p(image | history) = product over 1,024 pixels of
    Bernoulli(pixel; sigmoid(aligned plan logit + causal residual))
```

Each factor observes only earlier rows and earlier pixels in its row. This
gives an exact image NLL, evaluated in parallel during training. Generation
uses 32 row-transformer evaluations and 1,024 small masked-head calls per
glyph. Pixel values and spatial coordinates are sensor measurements, not
linguistic identities.

The inverse-image path copies supplied binary rasters exactly at initialization.
Motor exercises retain that path during training. Autonomous language requires
the model to choose the intended future image from the prompt; copying an
already-supplied answer does not demonstrate that ability.

## Data and training

Run from the repository root with the existing PyTorch environment and public
Chinese manifest at `data/visual_grammar/chinese_wikisource_public_domain.jsonl`.
The initialization checkpoints and fonts named in the protocol must exist.
Sources and data hashes are checked; completed runs are protected from
accidental replacement. Reference books are not required for this experiment.

V54 trained the density with a frozen reader. V55 compares continuation of
that frozen-reader training with joint reader/density training under the same
proper image likelihood. Both arms start from the identical V54 EMA. The
protocol fixes data draws, seeds, update counts, learning rates and audit
criteria before optimization.

```bash
# Sequential jobs on one physical GPU; finish both before the matched audit.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_joint_exact_raster_v55.py --arm frozen --device cuda:0
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_joint_exact_raster_v55.py --arm joint --device cuda:0
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_joint_exact_raster_v55.py --device cuda:0
```

Resume an interrupted arm with `--resume <arm>/checkpoint_latest.pt`. Model,
EMA, optimizer, RNGs and deterministic dataset offsets are saved. An exact
interrupted-versus-continuous smoke check precedes production training. The
joint arm's NLL gradient reaches the reader; reader dropout stays disabled in
both arms, and the frozen control receives no reader gradient.

## Independent image inference

The native prompt is a grayscale PNG containing an aligned grid of 32-pixel
cells, without gutters. Cells are read left to right, then top to bottom.
This interface does not yet segment an arbitrary book photograph or PDF page.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/infer_joint_exact_raster_v55.py \
  --prompt-image prompt.png --cells 16 --out answer.png \
  --export ilm_standalone.pt --device cuda:0

# The export embeds the complete reader and density, without the initialization files.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/infer_joint_exact_raster_v55.py \
  --checkpoint ilm_standalone.pt --prompt-image prompt.png \
  --cells 16 --out answer_again.png --device cuda:0
```

Use `--prompt-text '天地玄黃'` for host-side rendering, `--greedy` for deterministic
thresholding, or `--copy --greedy` for the explicitly answer-supplied motor test.
The same seed gives the same stochastic draw for the same checkpoint and
runtime. Each output includes the actual PNG, prompt PNG and a JSON boundary
receipt. No image-bank replacement or OCR repair is applied.

## Evidence and limits

Audits retain exact NLL, first samples, greedy outputs, visible feedback,
supplied-image controls, raw galleries, resource receipts and paired
source-record intervals. Evaluator banks are constructed after generation and
do not enter inference. The sealed source partition remains closed.

A low teacher-forced pixel loss can reflect predictable strokes within the
target. A clear glyph may still be the wrong answer. Qualification therefore
requires the registered context, surface, motor and autonomous-generation
tests together. Broad instruction following, etymology and historical-form
control need their own held-out data and measurements.
