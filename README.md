[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Paradigm](https://img.shields.io/badge/paradigm-predictive%20visual%20field-16835B)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Domain](https://img.shields.io/badge/domain-historic%20etymology%20%7C%20glyph%20models-2F80ED?logo=github)

![ILM-V image-native language model concept: image input to image output with 言 glyph evolution](publication/ilm-image-native/figures/ilm_v_yan_readme_hero.png)

*Image-native language modeling concept: a writing image enters ILM-V, the model reasons in visual latent space, and the answer is rendered as an image. The glyph panels use local hanziyuan-derived ziyuan data for the evolution of `言` (YAN, U+8A00).*

## Current Research Paradigm: Predictive Visual Field

![Predictive Visual Field: writing images become continuous retinal states, a causal field predicts the next visual state, a separate visual actuator writes it, and the generated pixels are reread](publication/ilm-image-native/figures/predictive_visual_field_paradigm.png)

The implemented RFLM V7 experiment exposed a structural error: one conditional
pixel flow was being asked to discover the next linguistic identity and render
its strokes in the same operation. The next falsifiable architecture separates
those jobs without relaxing the image-only boundary:

1. A retina learns a continuous manifold directly from writing images.
2. A causal fast/line/page field predicts a **distribution over the next visual
   state** with continuous flow matching.
3. A separate pixel-flow actuator renders a sampled visual state as ink.
4. The retina rereads the rendered pixels and feeds them back into the field.

There is no nearest-character lookup or output vocabulary. This Predictive
Visual Field is the V8 hypothesis, not a demonstrated capability. The first
proof must show that image-derived state flow beats last-only and unigram
baselines before the renderer or model is scaled.

The strict student boundary remains:

```text
writing pixels -> continuous visual dynamics -> continuous ink pixels
```

The student receives no strings, token IDs, Unicode IDs, OCR transcript,
character labels, external language model, or discrete visual codebook. Typed
input is supported only by deterministic rasterization before this boundary.
An uploaded page can enter directly as pixels.

### Implemented precursor: Retinal Flow V7

![Retinal Flow Language Model: ordered image fixations become a recurrent visual field, a rectified-flow writer generates candidate ink, and the candidates are reread and fed back](publication/ilm-image-native/figures/retinal_flow_paradigm.png)

The current runnable model is an 11.69M-parameter **Retinal Flow Language
Model**, a concrete
read-predict-write-reread loop:

1. A small convolutional retina reads ordered `32x32` grayscale fixations.
2. A three-layer recurrent visual field integrates the fixation history.
3. A continuous energy function scores arbitrary candidate images; it has no
   character output table.
4. A conditional rectified flow writes the next fixation directly in pixel
   space.
5. The model rereads its generated ink, selects a candidate by visual energy,
   and feeds those pixels back into the recurrent state.

### Measured status, not a capability claim

V7 kept the model at `11,690,244` parameters, added 800 updates on one RTX
4090, and generated `25.3` visual cells per second in its matched run. It added
normalized context advantage against independent image anchors and
backpropagation through sampled flow endpoints. V6 and V7 were tested on the
same 512 common Han characters, four font views, 2,423 eligible held-out
contexts, and frozen bank SHA-256.

| Gate | V6 closed loop | V7 selected step 5,800 | Interpretation |
|---|---:|---:|---|
| Retina oracle top-1 | `98.18%` | `98.27%` | Basic cross-font perception is not the main bottleneck. |
| Full-context top-1 | `1.20%` | **`2.31%`** | V7 beats last-only (`2.02%`) and unigram (`1.86%`), but not bigram (`13.58%`). |
| Normalized context log-probability gain | `-0.9066` | **`-0.2155`** | The calibrated deficit shrank by 76%, but full history still lowers mean target probability. |
| Generated context cosine gain | `+0.0077` | **`+0.0303`** | V7 passes the held-out generated-signal gate. |
| Late/early autonomous ink | `1.168` | `1.050` | Both loops keep nontrivial ink without late occupancy drift. |
| Sparse autonomous cells | `18.75%` | `15.63%` | V7 is denser, but its continuation is still unreadable. |

![Matched V6 and V7 autonomous comparison](publication/ilm-image-native/figures/anchor_identity_v7_result.png)

**Verdict: V7 is rejected as a language model.** It establishes a useful
training correction, not a complete language system. Raw target energy was
positive while normalized target probability was negative, proving that raw
score margins were an invalid acceptance measure. V7 does not prove readable
continuation, historical question answering, efficiency over a text LLM, or
Qwen-8B parity. The result motivates the Predictive Visual Field separation
shown above.

## Run The Retinal MVP

Build the provenance-bearing public-domain Chinese manifest:

```bash
PYTHONPATH=. python scripts/build_visual_grammar_manifest.py \
  --wikisource-root ../Books/resources/curated-books/chinese-classics/public-domain-canon \
  --out data/visual_grammar/chinese_wikisource_public_domain.jsonl
```

Train the current combined RFLM objective from scratch on one 24 GiB GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_retinal_flow_lm.py \
  --manifest data/visual_grammar/chinese_wikisource_public_domain.jsonl \
  --out artifacts/retinal_flow_chinese_anchor_identity \
  --sequence-length 48 \
  --energy-positions-per-sequence 8 \
  --batch-size 32 \
  --maximum-steps 6000 \
  --context-anchor-bank-size 512 \
  --context-anchor-views 4 \
  --context-advantage-weight 0.5 \
  --context-advantage-margin 0.5 \
  --sampled-identity-weight 0.2 \
  --sampled-identity-steps 2 \
  --rollout-start-step 800 \
  --rollout-ramp-steps 400 \
  --rollout-batch-size 8 \
  --rollout-steps 2 \
  --rollout-candidates 2 \
  --rollout-sample-steps 2 \
  --precision bf16
```

The exact measured V7 continuation command, frozen-bank receipt, and autonomous
comparison are recorded in
[`docs/retinal-flow-v7-anchor-identity-result.md`](docs/retinal-flow-v7-anchor-identity-result.md).

Run the strict fixed-bank evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_retinal_flow_lm.py \
  --checkpoint artifacts/retinal_flow_chinese_anchor_identity/checkpoint_latest.pt \
  --bank-size 512 \
  --prototype-views 4 \
  --evaluation-samples 3000 \
  --generation-contexts 192 \
  --out artifacts/retinal_flow_chinese_anchor_identity/fixed_glyph_bank
```

Generate an autonomous image continuation from typed or image input:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/infer_retinal_flow_lm.py \
  --checkpoint artifacts/retinal_flow_chinese_anchor_identity/checkpoint_latest.pt \
  --text '天地玄黃，宇宙洪荒。日月盈昃，辰宿列張。' \
  --new-cells 32 \
  --candidate-samples 8 \
  --out artifacts/retinal_flow_chinese_anchor_identity/autonomous_demo
```

The primary inference artifact is `complete_page.png`; `receipt.json` records
the model boundary, parameter count, throughput, VRAM, font hashes, every
candidate-selection step, and early/late autonomous trajectory summaries.
Generated checkpoints and data remain git-ignored.

The earlier whole-page U-Net, latent diffusion, associative-memory, and causal
InkStream implementations remain as baselines. They are not the current model.

ILM is a research codebase for **language learned and generated as visible
writing**. Its current experiment predicts continuous retinal states and writes
continuous ink with rectified flow. Older structured embeddings, codebooks, and
page diffusion experiments remain available as falsified or comparative
baselines; they do not define the current model boundary.

> The repository intentionally keeps a practical etymology pipeline and long-horizon ILM experimentation side-by-side.

## 📌 Overview

This repository has three connected tracks:

1. Retinal-flow image-native language modeling and strict held-out evaluation.
2. Historic Chinese glyph etymology ingestion and provenance-preserving assets.
3. Earlier glyph, codebook, diffusion, folio, and InkStream baselines retained
   for reproducibility.

This README documents all three tracks and keeps the etymology workflow as a first-class, reproducible path.

## 🔗 Key Links

| Area | Path |
|---|---|
| Conceptual write-up | `docs/imagized-language-model.md` |
| Current engineering goal | `docs/first-imagized-language-model-goal.md` |
| V7 anchor-identity experiment | `docs/retinal-flow-v7-anchor-identity-result.md` |
| Closed-loop V6 experiment | `docs/retinal-flow-v6-closed-loop-result.md` |
| Research dossier and evidence | `references/image-native-language-model-research.md` |
| Archived diffusion plan | `docs/ilm-visual-diffusion-code-plan.md` |
| Archived embedding "color" plan | `docs/embedding-color-plan.md` |
| Historical development plan | `docs/development-plan.md` |
| Etymology module readme | `ilm/etymology/README.md` |

## ✨ Features

- 🏺 Etymology ingestion from `hanziyuan` and `chineseetymology`-style sources.
- 👁️ Continuous foveal retina with recurrent visual context and cross-font invariance.
- 🖋️ Conditional pixel-space rectified-flow writer with a differentiable write-read cycle.
- 🔁 Autonomous image-only inference with candidate rereading, energy reranking, and pixel feedback.
- 🧭 Training on exact model-induced visual rollouts with state alignment, next-image energy, and recovery flow.
- 🧪 Fixed 512-character visual-bank evaluation against random, unigram, and bigram baselines.
- 🌐 Robust AJAX + HTML ingestion path with retries, throttling, and cache.
- 🧩 Stage-labeled glyph extraction including `<img>` and CSS `background-image` data URIs.
- 🗃️ SQLite-backed storage for chars/glyph metadata plus filesystem asset layout.
- 🖥️ Tornado web UI for ad-hoc ingest + gallery preview.
- 🔤 Glyph rendering utilities for multilingual token images.
- 🧠 Product-code style embedding/codebook modules.
- 🧱 Sentence frame packing and diffusion/inpainting training/evaluation scripts.
- 📊 Reporting and visualization scripts for embedding and pipeline inspection.
- 📄 Publication artifacts in LaTeX/PDF under `publication/`.

## 🧱 Project Structure

```text
.
├── README.md
├── AGENTS.md
├── configs/
│   ├── color.yaml
│   └── diffusion.yaml
├── docs/
├── i18n/
├── ilm/
│   ├── code/
│   ├── data/
│   ├── datasets/
│   ├── db/
│   ├── diffusion/
│   ├── encoders/
│   ├── english_tiles/
│   ├── etymology/
│   ├── frames/
│   ├── models/
│   ├── visual_lm/
│   └── utils/
├── scripts/
├── publication/
├── assets/
├── logs/
└── *.ipynb
```

## 🧰 Prerequisites

| Requirement | Notes |
|---|---|
| Python `3.10+` | Core runtime |
| `pip` | Package installation |
| Optional GPU | Helpful for PyTorch CUDA training scripts |
| Optional LaTeX toolchain | Needed for publication builds |

Assumption note: there is currently no single root dependency lock/spec file (`pyproject.toml`, `requirements.txt`, etc.), so dependencies are inferred from imports and script usage.

## ⚙️ Installation

### Minimal (etymology toolkit)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### Extended (modeling/training workflows)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch fonttools
```

If a specific script needs additional packages, install them from the import error shown by that script.

## 🚀 Usage

### Quick Start: Historic Glyph Ingestion (CLI)

1. Hanziyuan (recommended): char-only AJAX flow

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (direct URL)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. Batch file ingestion (lines can be `char\turl`, `url`, or `char url`)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### Outputs

| Output Type | Location |
|---|---|
| Files | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| Cache | `data/historic/cache/*.html` |
| DB | `data/historic/etymology.sqlite3` |

### Web Demo (optional)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

Open `http://127.0.0.1:8888`, choose site, enter a character (for example `中`).

### Polite Crawling and Site Respect

- The fetcher uses per-host throttling, retries with backoff, and caching.
- Keep delays `>= 0.5s`, avoid bursts, and honor site terms/robots/licensing.
- Do not bypass paywalls or interactive protections.
- If you see `403`/`429`, slow down and retry later.

### Additional ILM Workflows

These scripts exist and are actively part of the repo surface, but they are research workflows and may require prepared local datasets/checkpoints.

1. Data download/prep

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. Glyph DB lifecycle

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. Code/color model training

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. Diffusion/inpainting

```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_inpaint_frames.py --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt --out artifacts/inpaint
```

5. Evaluation/reporting

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## 🧩 Configuration

Primary YAML configs:

- `configs/color.yaml`
  - data path: `data/processed/images_common_freq/index.tsv`
  - model/code params: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - optimizer/log settings

- `configs/diffusion.yaml`
  - input JSONL: `data/processed/test_100.jsonl`
  - frame/grid + model size settings
  - train mask ratio range and checkpoint settings

Override settings via CLI flags where supported (`--epochs`, `--batch-size`, `--lr`, etc.).

## 🧪 Examples

- Build a single English tile glyph:

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- Run inpainting demo with trained checkpoints:

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- Bulk ingest common characters from Hanziyuan:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 📝 Development Notes

- This is a research repository with both robust CLIs and exploratory artifacts (including notebooks and prototype scripts).
- Generated large files are intended for `data/` and `artifacts/` (both ignored in `.gitignore`).
- Publication source and PDFs are under `publication/`; helper build script: `scripts/latex_build.sh`.
- Collaboration/process conventions are documented in `AGENTS.md`.

## 🛠️ Troubleshooting

- `ModuleNotFoundError: ilm...`
  - Run scripts from repo root.
  - Use `PYTHONPATH=.` for scripts that expect local package resolution.

- `FileNotFoundError` for data/index/checkpoints
  - Run prerequisite data/build scripts first.
  - Confirm defaults such as `data/processed/images_common_freq/index.tsv` and `data/processed/test_100.jsonl` exist.

- CUDA/device issues
  - Switch to CPU with script flags/config (`device: cpu` or `--device cpu`).

- Missing package errors
  - Install required dependency from the specific script import path (`torch`, `pyyaml`, `Pillow`, etc.).

- HTTP `403` / `429` while scraping
  - Increase `--delay`, retry later, and keep requests polite.

## 🗺️ Roadmap

- Implement continuous next-retina flow as the primary language distribution; keep pixel flow as a conditioned actuator.
- Add fast, line, and page visual states only through measured ablations, starting with the smallest causal state flow.
- Require full visual context to beat last-fixation, unigram, and bigram baselines.
- Require stable, readable 32-cell autonomous continuations before scaling width or corpus size.
- Add multiscale page memory and provenance-gated historical glyph composition only after the causal gate passes.
- Improve environment reproducibility with one authoritative dependency specification and focused tests.

For deeper conceptual and staged planning details, see:

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 🤝 Contributing

- Follow `AGENTS.md` for conventions (atomic commits, push after change, no credentials in code).
- Group related edits in focused commits with conventional messages.
- Prefer reproducible script invocations with explicit flags and input paths.
- For scraping-related changes, preserve throttling/cache behavior and site-respect constraints.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

No top-level license file is currently present in this repository.

Assumption note: treat the project as research code with unspecified licensing until a `LICENSE` file is added by maintainers.
