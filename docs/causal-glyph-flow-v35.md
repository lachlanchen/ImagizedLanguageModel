# Causal Glyph Flow V35

V35 is the repository's first production-sized attempt to join a causal visual
language backbone to a qualified continuous writing codec and close the loop on
the pixels it actually emits. It is a mechanism experiment, not a general LLM
or Qwen-parity claim.

The immutable design and decision rules are in:

- `references/causal_glyph_flow_v35_protocol.md`;
- `references/causal_glyph_flow_v35_research.md`; and
- `references/continuous_glyph_codec_v34_result.md`.

The measured V35 decision belongs in
`references/causal_glyph_flow_v35_result.md` after evaluation. A checkpoint or
training loss alone is not evidence of language behavior.

## Runtime Contract

The student model receives:

```text
pixels:     [batch, 1, 32, 32 * visual_patches]
patch_mask: [batch, visual_patches]
```

It returns binary `32 x 32` writing patches, generated visual lengths, visible
feedback latents, and stop probabilities. Its recurrent generation boundary is:

```text
predicted continuous latent
  -> frozen V34 decoder
  -> binary visible patch
  -> frozen V34 encoder
  -> next causal context
```

There is no token embedding, vocabulary output, Unicode or character ID,
visual codebook, OCR, retrieval key, or external-model call in this runtime
path. The command-line text option is only a deterministic boundary renderer;
the same inference command accepts a writing image directly. Optional OCR runs
after image generation and is labeled as evaluator-only.

## External Foundations

Using reliable external components is allowed and encouraged when provenance
and deployment boundaries remain explicit.

- V34 is this repository's frozen 7.4M-parameter codebook-free writing codec.
- The 12-layer causal field starts from the public PIXAR checkpoint. PIXAR's
  resized pixel projection is used only to align V34 coordinates and is then
  discarded.
- Noto CJK fonts render train, development, and sealed visual streams.
- Tesseract `chi_sim+chi_tra` scores generated PNGs outside the student.

The PIXAR source revision is MIT, but the downloaded weight archive does not
state a weight license. V35 artifacts are therefore local research evidence;
the exporter sets `weight_redistribution_authorized` to false.

## Data

The frozen route uses:

| Stream | Purpose | Rights |
| --- | --- | --- |
| Chinese Wikisource manifest | continuation and deterministic copy | public-domain source rows with retained provenance |
| short Chinese Alpaca subset | prompt-to-answer raster training | CC BY-NC 4.0; research only |
| fixed paraphrases | development wording-shift diagnostic | inherits Alpaca research-only rights |

The public split contains 6,743 train, 135 development, and 139 sealed source
records. The instruction split contains 5,747 train, 159 development, and 202
sealed records. Source identifiers are hash-partitioned before rendering.

## Training

Prerequisites are Python 3.10+, PyTorch with CUDA BF16 support, Transformers,
Pillow, NumPy, the four recorded Noto CJK fonts, the V34 checkpoint, and the
local PIXAR checkpoint.

The evidence workstation used Python 3.10, PyTorch `2.11.0+cu130`, CUDA 13.0,
Transformers 4.57.6, NumPy 2.2.6, and Pillow 10.4.0. Install a CUDA-compatible
PyTorch build through the official PyTorch channel first, then install the
remaining pinned packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-v35.txt
```

The evaluator additionally requires Tesseract 5 with `chi_sim` and `chi_tra`
traineddata. Rendering requires Noto Sans CJK Regular/Medium and Noto Serif CJK
Regular/Light at the paths recorded in the protocol.

```bash
PYTHONPATH=. python scripts/train_causal_glyph_flow_v35.py \
  --device cuda:0 \
  --out artifacts/causal_glyph_flow_v35_20260814
```

The evidence route is fixed at 2,000 adapter-alignment updates, 8,000 public
continuation updates, and 12,000 instruction/copy updates. Physical batch size
is eight with eight-way gradient accumulation. The model must remain below 20
GiB allocated VRAM on one RTX 4090.

An interrupted run resumes only when protocol, source, data, arguments, model,
optimizer, and RNG receipts match:

```bash
PYTHONPATH=. python scripts/train_causal_glyph_flow_v35.py \
  --resume artifacts/causal_glyph_flow_v35_20260814/checkpoint_latest.pt
```

Do not edit the five source files listed in the training run receipt while an
evidence run is active. New evaluators are intentionally outside that immutable
training source set.

## Development Decision

Install Tesseract with simplified and traditional Chinese traineddata, then run:

```bash
PYTHONPATH=. python scripts/eval_causal_glyph_flow_v35.py \
  --checkpoint artifacts/causal_glyph_flow_v35_20260814/checkpoint_latest.pt \
  --device cuda:0 \
  --out artifacts/causal_glyph_flow_v35_20260814/development
```

The evaluator audits raw and EMA states, teacher-forced next-patch quality,
anchor and eight-step Heun writers, target-through-codec OCR ceilings, and
closed raster generation. The selected writer then runs with correct,
cyclically shuffled, all-white, and final-quarter-only prompts. Copy pairs hold
font, size, origin, augmentation seed, prompt length, and target length fixed.

The status vocabulary is exactly:

- `not-qualified`;
- `visual-causal-qualified`; or
- `semantic-raster-qualified`.

Only EMA can decide the primary status. Raw weights are diagnostic. A
visual-causal result must not be described as semantic understanding.

## Sealed Evaluation

Do not run the sealed command manually unless development is at least
`visual-causal-qualified`. The script checks this before loading or rendering a
sealed record, locks the development-selected writer, and refuses an existing
sealed output directory:

```bash
PYTHONPATH=. python scripts/eval_causal_glyph_flow_v35_sealed.py \
  --checkpoint artifacts/causal_glyph_flow_v35_20260814/checkpoint_latest.pt \
  --development-report \
    artifacts/causal_glyph_flow_v35_20260814/development/development_report.json \
  --device cuda:0
```

The one-shot transfer decision requires every applicable absolute development
threshold and at least 90% retention of each primary metric. A failed sealed
result is retained and reported.

## Standalone Student

After development evaluation, export the selected student state:

```bash
PYTHONPATH=. python scripts/export_causal_glyph_flow_v35.py
```

The standalone file contains the V34 codec, aligned adapter, causal field,
anchor/flow writer, stop head, generation settings, and evidence receipts. It
contains no optimizer, RNG, alignment teacher, OCR, tokenizer, retrieval data,
or resumable training state. Export rejects an unqualified report unless
`--allow-unqualified` is explicitly supplied, in which case the diagnostic
status remains embedded in the artifact.

Typed prompt to PNG:

```bash
PYTHONPATH=. python scripts/infer_causal_glyph_flow_v35.py \
  --text '照写：天地 答：' \
  --out artifacts/v35_inference/tiandi.png
```

Image prompt to PNG:

```bash
PYTHONPATH=. python scripts/infer_causal_glyph_flow_v35.py \
  --image prompt_strip.png \
  --out artifacts/v35_inference/answer.png
```

Add `--ocr-sidecar` only when an external searchable transcription is useful.
The PNG remains the primary model output. V35 was trained on single-line binary
strips; accepting an arbitrary page image at the wrapper does not imply that
the current checkpoint understands page layout.
