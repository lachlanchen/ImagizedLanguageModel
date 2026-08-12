[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Paradigm](https://img.shields.io/badge/paradigm-predictive%20visual%20field-16835B)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Domain](https://img.shields.io/badge/domain-historic%20etymology%20%7C%20glyph%20models-2F80ED?logo=github)

ILM 研究如何把語言當作**可見書寫影像**來學習與生成：由影像進入連續視覺狀態，
再直接回到影像，不把隱藏的符號序列當作語言核心。

## 現行研究範式：預測視覺場

![預測視覺場架構](../publication/ilm-image-native/figures/predictive_visual_field_paradigm.png)

RFLM V7 證明了現行結構的一個關鍵問題：不應讓同一個像素 flow 同時推斷下一項
語言身分並完成筆畫渲染。V7 把完整上下文 top-1 從 `1.20%` 提升到 `2.31%`，
超過 last-only (`2.02%`) 與 unigram (`1.86%`)，但仍遠低於 bigram
(`13.58%`)。正規化目標對數機率增益從 `-0.9066` 改善到 `-0.2155`，但仍為
負值；32 格自主輸出仍不可讀。因此，**V7 作為語言模型未通過驗收**。

下一代 PVF 將「預測下一連續視網膜狀態」與「把該狀態繪製為墨跡並重新讀取」分開。
部署時沒有最近字元檢索，也沒有字元輸出表。嚴格邊界仍是：`書寫像素 -> 連續視覺
動力學 -> 墨跡像素`；學生不接收 token、Unicode ID、OCR、視覺碼本或外部語言
模型。PVF 是可證偽的 V8 假設，不是已實現的能力。

> 該儲存庫有意讓「可實際執行的語源工作流程」與「長程 ILM 實驗」保持並行。

## 📌 概覽

這個儲存庫目前有三條互相關聯的主線：

1. 視網膜流影像原生語言建模與嚴格的訓練外評估。
2. 保留來源證據的歷史漢字字形語源資料擷取。
3. 為重現而保留的早期字形、碼本、擴散、folio 與 InkStream 對照基線。

本 README 記錄三條主線，並將語源流程維持為可重現的主流程之一。

## 🔗 關鍵連結

| 項目 | 路徑 |
|---|---|
| 概念性說明 | `docs/imagized-language-model.md` |
| 現行工程目標 | `docs/first-imagized-language-model-goal.md` |
| 研究檔案與實驗證據 | `references/image-native-language-model-research.md` |
| 程式規劃與量測 | `docs/ilm-visual-diffusion-code-plan.md` |
| 嵌入「色彩」計畫 | `docs/embedding-color-plan.md` |
| 開發筆記與規劃 | `docs/development-plan.md` |
| 語源模組說明文件 | `ilm/etymology/README.md` |

## ✨ 功能

- 🏺 從 `hanziyuan` 與 `chineseetymology` 類型來源進行語源擷取。
- 🌐 穩健的 AJAX + HTML 擷取流程，具重試、節流與快取。
- 🧩 有階段標籤的字形擷取，包含 `<img>` 與 CSS `background-image` data URI。
- 🗃️ 使用 SQLite 儲存字元/字形中繼資料，搭配檔案系統資產配置。
- 🖥️ Tornado 網頁介面，支援臨時擷取與圖庫預覽。
- 🔤 多語系 token 影像的字形渲染工具。
- 🧠 product-code 風格的嵌入/碼本模組。
- 🧱 句子 frame 打包與 diffusion / inpainting 訓練與評估腳本。
- 📊 嵌入與流程檢視用的報告與視覺化腳本。
- 📄 `publication/` 下的 LaTeX/PDF 發表成果。

## 🧱 專案結構

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

## 🧰 先決條件

| 項目 | 說明 |
|---|---|
| Python `3.10+` | 核心執行環境 |
| `pip` | 套件安裝 |
| 可選 GPU | 有助於 PyTorch CUDA 訓練腳本 |
| 可選 LaTeX 工具鏈 | 發表建置所需 |

注意：目前尚未有單一的根目錄相依性鎖定/規格檔（如 `pyproject.toml`、`requirements.txt`），因此相依性是從匯入與腳本使用情境中推斷。

## ⚙️ 安裝

### 最小安裝（語源工具鏈）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### 擴充安裝（建模與訓練工作流程）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

若某個腳本需要額外套件，請依該腳本回報的 import error 再安裝。

## 🚀 使用

### 快速開始：歷史字形擷取（CLI）

1. Hanziyuan（建議）：僅字元 AJAX 流程

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology（直接 URL）

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. 批次檔案擷取（每行可為 `char\turl`、`url`，或 `char url`）

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### 輸出

| 輸出類型 | 位置 |
|---|---|
| 檔案 | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| 快取 | `data/historic/cache/*.html` |
| 資料庫 | `data/historic/etymology.sqlite3` |

### 網頁示範（可選）

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

開啟 `http://127.0.0.1:8888`，選擇網站並輸入一個字元（例如 `中`）。

### 禮貌爬取與網站尊重

- 擷取器使用每個主機的節流、帶退避的重試與快取。
- 請保留 `>= 0.5s` 的延遲、避免突發流量，並遵守網站條款與 robots / 授權規範。
- 勿繞過付費牆或互動式保護機制。
- 若出現 `403` / `429`，請放慢速度並稍後重試。

### 其他 ILM 工作流程

這些腳本目前皆在版本庫中，且仍屬研究流程，可能需要事先準備本地資料集與檢查點。

1. 資料下載與預處理

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. 字形資料庫生命週期

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. 代碼/色彩模型訓練

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. Diffusion / inpainting

```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_inpaint_frames.py --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt --out artifacts/inpaint
```

5. 評估 / 報告

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## 🧩 組態

主要 YAML 設定檔：

- `configs/color.yaml`
  - 資料路徑：`data/processed/images_common_freq/index.tsv`
  - 模型 / 代碼參數：`d_glyph`、`d_code`、`K`、`C`、temperature/anneal
  - optimizer / log 設定

- `configs/diffusion.yaml`
  - 輸入 JSONL：`data/processed/test_100.jsonl`
  - frame / grid 與模型大小設定
  - 訓練遮罩比例範圍與 checkpoint 設定

可在支援 CLI 的情況下使用旗標覆寫設定（`--epochs`、`--batch-size`、`--lr` 等）。

## 🧪 範例

- 建立單一英文字 token 字形：

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- 使用已訓練 checkpoint 執行 inpainting 示範：

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

## 📝 開發備註

- 這是個研究型版本庫，兼具穩定可用的 CLI 與探索性產物（包含 notebooks 與原型腳本）。
- 產生的大型檔案預期放在 `data/` 與 `artifacts/`（兩者皆在 `.gitignore` 中）。
- 發表來源與 PDF 位於 `publication/`；輔助建置腳本為 `scripts/latex_build.sh`。
- 協作與流程規範記載於 `AGENTS.md`。

## 🛠️ 除錯

- `ModuleNotFoundError: ilm...`
  - 從版本庫根目錄執行腳本。
  - 對需要本地套件解析的腳本使用 `PYTHONPATH=.`。

- `FileNotFoundError`（資料 / index / checkpoint）
  - 先執行先決資料／建置腳本。
  - 確認預設檔案存在，例如 `data/processed/images_common_freq/index.tsv` 與 `data/processed/test_100.jsonl`。

- CUDA / 裝置問題
  - 透過腳本旗標或設定改為 CPU（`device: cpu` 或 `--device cpu`）。

- 缺少套件
  - 依腳本的 import 路徑安裝所需套件（如 `torch`、`pyyaml`、`Pillow`）。

- 爬取時出現 HTTP `403` / `429`
  - 提高 `--delay`，稍後再試，並維持禮貌請求。

## 🗺️ 路線圖

- 繼續將文字即影像 ILM 的訓練／評估 runbook 從以語源為先的快速開始擴展。
- 改善環境重現性（建立單一權威相依性規格）。
- 擴大研究腳本與 pipeline glue 的測試與 CI 覆蓋。
- 持續優化階層式碼本、diffusion 目標與可控性通道。
- 彙整 `docs/`、腳本說明與發表物件中的文件。

若需更深入的概念與分階段規劃細節，請參考：

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 🤝 貢獻

- 依 `AGENTS.md` 的規範操作（原子提交、每次變更後推送、程式碼中不包含憑證）。
- 將相關修改集中到聚焦提交，並使用慣用提交訊息。
- 優先使用可重現的腳本呼叫，並明確指定旗標與輸入路徑。
- 若修改關係到爬取，請保留節流 / 快取機制與網站尊重限制。

## 📄 授權

目前這個版本庫尚未放置頂層授權檔。

注意：在維護者補上 `LICENSE` 檔前，請將此專案視為尚未明確授權的研究程式碼。


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
