[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

ILM 是一个研究型代码库，探索“将文本视作图像生成”的方法：它将语言编码为紧凑、类图像的张量，并通过扩散式迭代细化来生成文本。该表示会将句子分解为元要素（语法、语义、语气、情绪），以及面向词和字符的分层、类记忆编码。它统一了离散扩散、叠加/解耦、结构化嵌入与字形感知字符建模等思路。

## 概览

当前仓库包含两条主要的实践路线：

1. 历史汉字字形词源数据摄取（抓取/解析/存储/预览）。
2. ILM 字形/图像建模实验（token 字形渲染、乘积码本、帧打包、扩散/修补、评估/报告）。

本仓库当前 README 在历史上以词源工具链为中心。该工作流在下文仍完整保留并作为规范基线。

## 关键链接

| 领域 | 路径 |
|---|---|
| 概念说明文档 | `docs/imagized-language-model.md` |
| 代码计划与指标 | `docs/ilm-visual-diffusion-code-plan.md` |
| 嵌入“颜色”计划 | `docs/embedding-color-plan.md` |
| 开发笔记/计划 | `docs/development-plan.md` |
| 词源模块说明 | `ilm/etymology/README.md` |

## 功能特性

- 🏺 从 `hanziyuan` 与 `chineseetymology` 风格来源摄取词源数据。
- 🌐 稳健的 AJAX + HTML 摄取路径，包含重试、限速与缓存。
- 🧩 带阶段标签的字形提取，支持 `<img>` 与 CSS `background-image` 的 data URI。
- 🗃️ 基于 SQLite 存储字符/字形元数据，并配套文件系统资源布局。
- 🖥️ 基于 Tornado 的 Web UI，用于临时摄取与图库预览。
- 🔤 用于多语言 token 图像的字形渲染工具。
- 🧠 乘积码风格的嵌入/码本模块。
- 🧱 句子帧打包与扩散/修补训练、评估脚本。
- 📊 用于嵌入与流水线检查的报告和可视化脚本。
- 📄 `publication/` 下的 LaTeX/PDF 论文产物。

## 项目结构

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

## 前置要求

| 要求 | 说明 |
|---|---|
| Python `3.10+` | 核心运行环境 |
| `pip` | 包安装工具 |
| 可选 GPU | 对 PyTorch CUDA 训练脚本有帮助 |
| 可选 LaTeX 工具链 | 构建论文产物所需 |

假设说明：当前仓库根目录暂无统一依赖锁定/规范文件（`pyproject.toml`、`requirements.txt` 等），因此依赖主要从导入项与脚本用法推断。

## 安装

### 最小安装（词源工具链）

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

如果某个脚本需要额外包，请根据该脚本报出的 import 错误进行安装。

## 使用

### 快速开始：历史字形词源摄取（CLI）

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

### 礼貌抓取与站点尊重

- 抓取器使用按主机限速、带退避的重试和缓存。
- 保持延迟 `>= 0.5s`，避免突发请求，并遵守站点条款/robots/许可证。
- 不要绕过付费墙或交互式防护。
- 如遇 `403`/`429`，请放慢速度并稍后重试。

### 其他 ILM 工作流

这些脚本已存在并且是仓库活跃功能面的一部分，但它们属于研究工作流，可能需要预先准备本地数据集/检查点。

1. 数据下载/预处理

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

3. 码/颜色模型训练

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. 扩散/修补

```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_inpaint_frames.py --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt --out artifacts/inpaint
```

5. 评估/报告

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## 配置

主要 YAML 配置：

- `configs/color.yaml`
  - 数据路径：`data/processed/images_common_freq/index.tsv`
  - 模型/编码参数：`d_glyph`、`d_code`、`K`、`C`、temperature/anneal
  - 优化器/日志设置

- `configs/diffusion.yaml`
  - 输入 JSONL：`data/processed/test_100.jsonl`
  - 帧/网格与模型尺寸设置
  - 训练掩码比例范围与检查点设置

在支持的情况下，可通过 CLI 参数覆盖设置（`--epochs`、`--batch-size`、`--lr` 等）。

## 示例

- 构建单个英文 tile 字形：

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- 使用已训练检查点运行修补演示：

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- 从 Hanziyuan 批量摄取常用字：

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 开发说明

- 这是一个研究型仓库，既包含稳健的 CLI，也包含探索性产物（包括 notebooks 和原型脚本）。
- 生成的大文件建议放在 `data/` 与 `artifacts/`（两者均在 `.gitignore` 中忽略）。
- 论文源文件与 PDF 位于 `publication/`；辅助构建脚本：`scripts/latex_build.sh`。
- 协作/流程约定记录在 `AGENTS.md`。

## 故障排查

- `ModuleNotFoundError: ilm...`
  - 从仓库根目录运行脚本。
  - 对需要本地包解析的脚本使用 `PYTHONPATH=.`。

- 数据/索引/检查点 `FileNotFoundError`
  - 先运行前置数据/构建脚本。
  - 确认默认路径（如 `data/processed/images_common_freq/index.tsv` 和 `data/processed/test_100.jsonl`）存在。

- CUDA/设备问题
  - 通过脚本参数/配置切换到 CPU（`device: cpu` 或 `--device cpu`）。

- 缺少包错误
  - 按具体脚本的 import 路径安装所需依赖（`torch`、`pyyaml`、`Pillow` 等）。

- 抓取时出现 HTTP `403` / `429`
  - 增大 `--delay`、稍后重试，并保持请求礼貌。

## 路线图

- 在词源优先的快速开始之外，继续完善 text-as-image ILM 训练/评估操作手册。
- 提升环境可复现性（单一权威依赖规范）。
- 扩展研究脚本与流水线胶水代码的测试/CI 覆盖率。
- 迭代分层码本、扩散目标与可控性通道。
- 统一 `docs/`、脚本帮助文本与论文产物之间的文档。

更多概念与分阶段规划细节，请参见：

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 贡献

- 请遵循 `AGENTS.md` 中的约定（原子提交、修改后推送、代码中不包含凭据）。
- 将相关改动分组为聚焦提交，并使用约定式提交消息。
- 优先使用带显式参数和输入路径、可复现的脚本调用方式。
- 对抓取相关改动，请保留限速/缓存行为和站点尊重约束。

## 许可证

当前仓库顶层尚不存在许可证文件。

假设说明：在维护者添加 `LICENSE` 文件前，请将本项目视为许可证未指定的研究代码。
