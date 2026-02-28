[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

ILM 是一個研究型程式碼庫，探索「文字即影像（text-as-image）」生成：它將語言編碼為緊湊、類影像的張量，並以 diffusion 風格的反覆精煉來生成文字。此表徵將句子拆解為後設元素（文法、語意、語氣、情緒）與具層級、類記憶的詞彙與字元編碼，整合了離散 diffusion、superposition/disentanglement、結構化嵌入，以及具字形感知的字元建模等想法。

## 概覽

此儲存庫目前包含兩條主要且可實作的路線：

1. 歷史漢字字形語源資料擷取（爬取/解析/儲存/預覽）。
2. ILM 字形/影像建模實驗（token 字形渲染、乘積碼本、frame 打包、diffusion/inpainting、評估/報告）。

本儲存庫目前的 README 在歷史上以語源工具鏈為核心。該工作流程仍在下文完整記錄，並保留為正典內容。

## 重要連結

| 區域 | 路徑 |
|---|---|
| 概念說明文件 | `docs/imagized-language-model.md` |
| 程式計畫與指標 | `docs/ilm-visual-diffusion-code-plan.md` |
| 嵌入「色彩」計畫 | `docs/embedding-color-plan.md` |
| 開發筆記/計畫 | `docs/development-plan.md` |
| 語源模組說明 | `ilm/etymology/README.md` |

## 功能

- 🏺 從 `hanziyuan` 與 `chineseetymology` 類型來源進行語源資料擷取。
- 🌐 穩健的 AJAX + HTML 擷取流程，含重試、節流與快取。
- 🧩 具 stage 標記的字形抽取，包含 `<img>` 與 CSS `background-image` data URI。
- 🗃️ 以 SQLite 儲存字元/字形中繼資料，並搭配檔案系統資產佈局。
- 🖥️ 以 Tornado 提供臨時擷取 + 圖庫預覽 Web UI。
- 🔤 多語 token 影像的字形渲染工具。
- 🧠 乘積碼（product-code）風格的嵌入/碼本模組。
- 🧱 句子 frame 打包與 diffusion/inpainting 訓練/評估腳本。
- 📊 用於嵌入與流程檢視的報告與視覺化腳本。
- 📄 `publication/` 目錄下的 LaTeX/PDF 發表產物。

## 專案結構

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

## 先決條件

| 需求 | 說明 |
|---|---|
| Python `3.10+` | 核心執行環境 |
| `pip` | 套件安裝 |
| 可選 GPU | 對 PyTorch CUDA 訓練腳本有幫助 |
| 可選 LaTeX 工具鏈 | 建置 publication 時需要 |

假設說明：目前尚無單一的根目錄依賴鎖定/規格檔（如 `pyproject.toml`、`requirements.txt` 等），因此依賴是由匯入與腳本使用情境推斷。

## 安裝

### 最小安裝（語源工具鏈）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### 擴充安裝（建模/訓練工作流程）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

若特定腳本需要其他套件，請依該腳本顯示的 import error 補安裝。

## 使用方式

### 快速開始：歷史字形擷取（CLI）

1. Hanziyuan（建議）：僅字元 AJAX 流程

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology（直接 URL）

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. 批次檔案擷取（每行可為 `char\turl`、`url` 或 `char url`）

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### 輸出

| 輸出類型 | 位置 |
|---|---|
| 檔案 | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| 快取 | `data/historic/cache/*.html` |
| DB | `data/historic/etymology.sqlite3` |

### Web 示範（可選）

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

開啟 `http://127.0.0.1:8888`，選擇網站並輸入字元（例如 `中`）。

### 禮貌爬取與網站尊重

- 擷取器使用每個 host 的節流、含 backoff 的重試，以及快取。
- 請保持延遲 `>= 0.5s`、避免突發請求，並遵守站點條款/robots/授權。
- 不要繞過付費牆或互動式保護。
- 若出現 `403`/`429`，請放慢速度並稍後再試。

### 其他 ILM 工作流程

這些腳本確實存在，且是目前儲存庫的一部分；但它們屬於研究型工作流程，可能需要預先準備的本地資料集/檢查點。

1. 資料下載/前處理

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. 字形 DB 生命週期

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. code/color 模型訓練

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. diffusion/inpainting

```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_inpaint_frames.py --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt --out artifacts/inpaint
```

5. 評估/報告

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## 設定

主要 YAML 設定檔：

- `configs/color.yaml`
  - 資料路徑：`data/processed/images_common_freq/index.tsv`
  - 模型/code 參數：`d_glyph`、`d_code`、`K`、`C`、temperature/anneal
  - optimizer/log 設定

- `configs/diffusion.yaml`
  - 輸入 JSONL：`data/processed/test_100.jsonl`
  - frame/grid 與模型尺寸設定
  - 訓練遮罩比例範圍與 checkpoint 設定

可在支援的情況下以 CLI 旗標覆寫設定（`--epochs`、`--batch-size`、`--lr` 等）。

## 範例

- 建立單一英文字 tile 字形：

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- 使用訓練好的 checkpoints 執行 inpainting 示範：

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- 從 Hanziyuan 批次擷取常用字：

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 開發備註

- 這是研究型儲存庫，同時包含穩健 CLI 與探索性產物（含 notebooks 與原型腳本）。
- 產生的大型檔案預期放在 `data/` 與 `artifacts/`（兩者皆在 `.gitignore` 中）。
- 發表來源與 PDF 位於 `publication/`；輔助建置腳本：`scripts/latex_build.sh`。
- 協作/流程慣例記錄於 `AGENTS.md`。

## 疑難排解

- `ModuleNotFoundError: ilm...`
  - 請從 repo root 執行腳本。
  - 對預期使用本地套件解析的腳本，請使用 `PYTHONPATH=.`。

- 資料/index/checkpoints 的 `FileNotFoundError`
  - 先執行必要的資料/建置腳本。
  - 確認預設路徑存在，例如 `data/processed/images_common_freq/index.tsv` 與 `data/processed/test_100.jsonl`。

- CUDA/裝置問題
  - 透過腳本旗標/設定改用 CPU（`device: cpu` 或 `--device cpu`）。

- 套件缺失錯誤
  - 依特定腳本的 import 路徑安裝所需依賴（`torch`、`pyyaml`、`Pillow` 等）。

- 擷取時遇到 HTTP `403` / `429`
  - 增加 `--delay`、稍後重試，並維持禮貌請求頻率。

## 路線圖

- 持續完善文字即影像 ILM 的訓練/評估 runbook，不再僅限於語源優先的快速開始。
- 改善環境可重現性（單一且權威的依賴規格）。
- 擴大研究腳本與 pipeline glue 的測試/CI 覆蓋率。
- 迭代層級式碼本、diffusion 目標與可控性通道。
- 整併 `docs/`、腳本說明文字與 publication 產物中的文件。

如需更深入的概念與分階段規劃細節，請參閱：

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 貢獻

- 請依 `AGENTS.md` 規範（原子提交、每次變更後推送、程式碼中不含憑證）。
- 以慣例訊息將相關修改分組為聚焦提交。
- 優先使用可重現的腳本呼叫，並明確指定旗標與輸入路徑。
- 若涉及爬取相關修改，請保留節流/快取行為與網站尊重限制。

## 授權

目前此儲存庫頂層尚未提供授權檔。

假設說明：在維護者新增 `LICENSE` 檔前，請將本專案視為授權未明確的研究程式碼。
