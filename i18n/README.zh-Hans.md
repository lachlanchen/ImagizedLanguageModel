[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Domain](https://img.shields.io/badge/domain-historic%20etymology%20%7C%20glyph%20models-2F80ED?logo=github)

ILM 是一个研究代码库，探索**文字即影像（text-as-image）生成**：它将语言编码为紧凑的类影像张量，并通过类扩散（diffusion）的迭代细化方式生成文本。该表征把句子拆解为元元素（语法、语义、语气、情绪）以及具有层级结构、类记忆特征的词汇与字符码。它融合了离散 diffusion、superposition/disentanglement、结构化 embedding，以及字形感知的字符建模思路。

> 该仓库有意将可复现的语源流程与面向长跨度的 ILM 实验并置。

## 📌 概览

该仓库目前有两条并行的主线：

1. 历史中文字形语源数据摄取（爬取/解析/存储/预览）。
2. ILM 字形/图像建模实验（token 字形渲染、码本、frame 打包、diffusion/inpainting、评估与报告）。

本 README 同时记录这两条主线，并将语源流程作为一等公民、可复现的路线保留。

## 🔗 关键链接

| 区域 | 路径 |
|---|---|
| 概念文档 | `docs/imagized-language-model.md` |
| 代码计划与指标 | `docs/ilm-visual-diffusion-code-plan.md` |
| Embedding "color" 计划 | `docs/embedding-color-plan.md` |
| 开发说明/计划 | `docs/development-plan.md` |
| 语源模块 readme | `ilm/etymology/README.md` |

## ✨ 特性

- 🏺 来自 `hanziyuan` 与 `chineseetymology` 风格来源的语源数据摄取。
- 🌐 稳健的 AJAX + HTML 摄取流程，带有重试、节流与缓存。
- 🧩 分阶段标注的字形提取，包含 `<img>` 与 CSS `background-image` data URI。
- 🗃️ SQLite 支持的字符/字形元数据存储，以及文件系统资源布局。
- 🖥️ Tornado Web UI，可进行临时摄取与图库预览。
- 🔤 多语言 token 图像的字形渲染工具。
- 🧠 Product-code 风格的 embedding/码本模块。
- 🧱 句子 frame 打包与 diffusion/inpainting 训练与评估脚本。
- 📊 嵌入与流程检查的报告与可视化脚本。
- 📄 `publication/` 下的 LaTeX/PDF 发布素材。

## 🧱 项目结构

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
│   └── utils/
├── scripts/
├── publication/
├── assets/
├── logs/
└── *.ipynb
```

## 🧰 先决条件

| 要求 | 说明 |
|---|---|
| Python `3.10+` | 核心运行时 |
| `pip` | 安装依赖 |
| 可选 GPU | 有助于运行 PyTorch CUDA 训练脚本 |
| 可选 LaTeX 工具链 | 用于构建发布物 |

假设说明：目前尚无单一的根级依赖锁定/规范文件（如 `pyproject.toml`、`requirements.txt` 等），因此依赖关系需根据 import 和脚本使用推断。

## ⚙️ 安装

### 最小安装（语源工具链）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### 扩展安装（建模/训练工作流）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

若某个具体脚本需要额外依赖，请按该脚本报出的导入错误逐一安装。

## 🚀 使用方式

### 快速开始：历史字形摄取（CLI）

1. Hanziyuan（推荐）：仅字符 AJAX 流程

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology（直接 URL）

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. 批量文件摄取（每行可为 `char\turl`、`url` 或 `char url`）

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### 输出

| 输出类型 | 位置 |
|---|---|
| 文件 | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| 缓存 | `data/historic/cache/*.html` |
| 数据库 | `data/historic/etymology.sqlite3` |

### Web 演示（可选）

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

打开 `http://127.0.0.1:8888`，选择站点并输入一个字符（例如 `中`）。

### 礼貌抓取与站点约束

- 抓取器使用按主机节流、带退避重试与缓存。
- 保持延迟 `>= 0.5s`，避免突发请求，并遵守站点条款/robots/授权约束。
- 不要绕过付费墙或交互式保护。
- 如果你看到 `403`/`429`，请放慢速率后稍后重试。

### 额外的 ILM 工作流

这些脚本均已存在并是仓库可见面向研究的流程的一部分，但它们是研究流程，可能需要本地预先准备的数据集/检查点。

1. 数据下载与预处理

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. 字形数据库生命周期

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. Code/color 模型训练

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

5. 评估与报告

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## 🧩 配置

主要 YAML 配置文件：

- `configs/color.yaml`
  - 数据路径：`data/processed/images_common_freq/index.tsv`
  - 模型/code 参数：`d_glyph`、`d_code`、`K`、`C`、temperature/anneal
  - optimizer/log 设置

- `configs/diffusion.yaml`
  - 输入 JSONL：`data/processed/test_100.jsonl`
  - frame/grid 与模型规模设置
  - 训练掩码比例范围与 checkpoint 设置

在支持的情况下，可用 CLI 标志覆盖配置（如 `--epochs`、`--batch-size`、`--lr`）。

## 🧪 示例

- 构建单个英语 tile 字形：

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- 使用训练完成的检查点运行 inpainting 演示：

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- 从 Hanziyuan 批量摄取常见字符：

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 📝 开发备注

- 这是一个研究型仓库，同时包含稳健 CLI 与探索性成果（包括 notebook 与原型脚本）。
- 大文件输出应写入 `data/` 和 `artifacts/`（二者均在 `.gitignore` 中）。
- 发布源码与 PDF 位于 `publication/`；辅助构建脚本为 `scripts/latex_build.sh`。
- 协作与流程约定记录在 `AGENTS.md`。

## 🛠️ 故障排查

- `ModuleNotFoundError: ilm...`
  - 从仓库根目录运行脚本。
  - 对需要本地包解析的脚本使用 `PYTHONPATH=.`。

- 数据/index/checkpoints 的 `FileNotFoundError`
  - 先运行前置的数据/构建脚本。
  - 确认默认项存在，例如 `data/processed/images_common_freq/index.tsv` 与 `data/processed/test_100.jsonl`。

- CUDA/设备问题
  - 用脚本参数/配置改为 CPU（`device: cpu` 或 `--device cpu`）。

- 缺少依赖错误
  - 按照具体脚本的 import 路径安装所需依赖（`torch`、`pyyaml`、`Pillow` 等）。

- 抓取时出现 HTTP `403` / `429`
  - 提高 `--delay`，稍后重试，并保持抓取行为礼貌。

## 🗺️ 路线图

- 在语源优先快速起步之外，继续完善 text-as-image ILM 的训练/评估 runbook。
- 改进环境可复现性（统一、权威的依赖规范）。
- 扩展研究脚本与 pipeline glue 的测试/CI 覆盖。
- 在分层码本、diffusion 目标与可控通道上持续迭代。
- 统一 `docs/`、脚本帮助文本与发布素材中的文档。

更深入的概念与分阶段规划细节可参见：

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 🤝 贡献

- 按 `AGENTS.md` 的约定执行（原子提交、变更后推送、代码中不包含凭证）。
- 将相关修改聚合为聚焦提交，并使用规范化提交信息。
- 优先使用可复现的脚本调用，明确传入标志和输入路径。
- 进行抓取相关修改时，保留节流/缓存行为及站点尊重约束。

## 📄 许可证

仓库当前未包含顶层许可证文件。

假设说明：在维护者添加 `LICENSE` 文件前，请将该项目按未明确指定许可证的研究代码对待。


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
