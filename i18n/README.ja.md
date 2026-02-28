[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

ILM は、text-as-image 生成を探究する研究コードベースです。言語をコンパクトな画像ライクなテンソルへ符号化し、拡散モデル風の反復的な洗練によってテキストを生成します。この表現は文をメタ要素（文法、意味、トーン、感情）と、単語・文字のための階層的なメモリライクコードへ分解します。これにより、離散拡散、重ね合わせ/分離表現、構造化埋め込み、グリフを意識した文字モデリングの発想を統合します。

## 概要

このリポジトリには現在、実用的な大きなトラックが 2 つあります。

1. 歴史的な漢字字形語源データの取り込み（スクレイピング/パース/保存/プレビュー）。
2. ILM のグリフ/画像モデリング実験（トークングリフ描画、積コードブック、フレームパッキング、拡散/インペインティング、評価/レポーティング）。

このリポジトリの README は歴史的に語源ツールキットを中心に構成されてきました。以下には、そのワークフローを正規情報として完全に維持したまま記載しています。

## 主要リンク

| 領域 | パス |
|---|---|
| コンセプト解説 | `docs/imagized-language-model.md` |
| コード計画と指標 | `docs/ilm-visual-diffusion-code-plan.md` |
| 埋め込み「色」計画 | `docs/embedding-color-plan.md` |
| 開発ノート/計画 | `docs/development-plan.md` |
| 語源モジュール README | `ilm/etymology/README.md` |

## 特徴

- 🏺 `hanziyuan` および `chineseetymology` 系ソースからの語源データ取り込み。
- 🌐 リトライ、スロットリング、キャッシュを備えた堅牢な AJAX + HTML 取り込み経路。
- 🧩 `<img>` と CSS `background-image` data URI を含む、ステージラベル付きグリフ抽出。
- 🗃️ 文字/グリフメタデータとファイルシステム資産レイアウトのための SQLite ベース保存。
- 🖥️ アドホック取り込みとギャラリープレビュー向け Tornado Web UI。
- 🔤 多言語トークン画像向けのグリフ描画ユーティリティ。
- 🧠 product-code 方式の埋め込み/コードブックモジュール。
- 🧱 文フレームパッキングと拡散/インペインティングの学習・評価スクリプト。
- 📊 埋め込みおよびパイプライン検証のためのレポート/可視化スクリプト。
- 📄 `publication/` 配下の LaTeX/PDF 公開成果物。

## プロジェクト構成

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

## 前提要件

| 要件 | 備考 |
|---|---|
| Python `3.10+` | コア実行環境 |
| `pip` | パッケージインストール |
| 任意の GPU | PyTorch CUDA 学習スクリプトで有用 |
| 任意の LaTeX ツールチェーン | 出版物ビルドに必要 |

前提メモ: 現時点では単一のルート依存ロック/定義ファイル（`pyproject.toml`, `requirements.txt` など）がないため、依存関係は import とスクリプト利用状況から推定されます。

## インストール

### 最小構成（語源ツールキット）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### 拡張構成（モデリング/学習ワークフロー）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

特定スクリプトで追加パッケージが必要な場合は、そのスクリプトの import エラーに従って追加インストールしてください。

## 使い方

### クイックスタート: 歴史的グリフ取り込み（CLI）

1. Hanziyuan（推奨）: 文字のみの AJAX フロー

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology（直接 URL）

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. バッチファイル取り込み（各行は `char\turl`、`url`、`char url` のいずれか）

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### 出力

| 出力タイプ | 保存先 |
|---|---|
| ファイル | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| キャッシュ | `data/historic/cache/*.html` |
| DB | `data/historic/etymology.sqlite3` |

### Web デモ（任意）

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

`http://127.0.0.1:8888` を開き、サイトを選択して文字（例: `中`）を入力します。

### 礼節あるクロールとサイト尊重

- フェッチャーはホスト単位のスロットリング、バックオフ付きリトライ、キャッシュを使用します。
- 遅延は `>= 0.5s` を維持し、バーストアクセスを避け、サイト規約/robots/ライセンスを尊重してください。
- ペイウォールや対話型保護の回避は行わないでください。
- `403`/`429` が出た場合は速度を落として後で再試行してください。

### 追加の ILM ワークフロー

これらのスクリプトは存在し、リポジトリの有効な構成要素です。ただし研究用ワークフローのため、ローカルで準備済みのデータセット/チェックポイントが必要な場合があります。

1. データ取得/前処理

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. Glyph DB ライフサイクル

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. コード/色モデル学習

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. 拡散/インペインティング

```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_inpaint_frames.py --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt --out artifacts/inpaint
```

5. 評価/レポーティング

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## 設定

主要な YAML 設定:

- `configs/color.yaml`
  - データパス: `data/processed/images_common_freq/index.tsv`
  - モデル/コード関連パラメータ: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - optimizer/log 設定

- `configs/diffusion.yaml`
  - 入力 JSONL: `data/processed/test_100.jsonl`
  - フレーム/グリッド + モデルサイズ設定
  - 学習マスク比率レンジとチェックポイント設定

対応している場合は CLI フラグ（`--epochs`, `--batch-size`, `--lr` など）で設定を上書きできます。

## 例

- 単一の英語タイルグリフを作成:

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- 学習済みチェックポイントでインペインティングデモを実行:

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- Hanziyuan から頻出文字を一括取り込み:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 開発ノート

- このリポジトリは、堅牢な CLI と探索的成果物（ノートブックやプロトタイプスクリプトを含む）を併せ持つ研究リポジトリです。
- 生成される大きなファイルは `data/` と `artifacts/` に置く想定です（どちらも `.gitignore` で無視）。
- 出版物のソースと PDF は `publication/` 配下にあります。補助ビルドスクリプト: `scripts/latex_build.sh`。
- コラボレーション/プロセス規約は `AGENTS.md` に記載されています。

## トラブルシューティング

- `ModuleNotFoundError: ilm...`
  - リポジトリルートからスクリプトを実行してください。
  - ローカルパッケージ解決を前提とするスクリプトでは `PYTHONPATH=.` を使用してください。

- データ/インデックス/チェックポイントに対する `FileNotFoundError`
  - 先に前提となるデータ作成スクリプトを実行してください。
  - `data/processed/images_common_freq/index.tsv` や `data/processed/test_100.jsonl` などのデフォルトファイルが存在することを確認してください。

- CUDA/デバイス関連の問題
  - スクリプトフラグ/設定（`device: cpu` または `--device cpu`）で CPU 実行に切り替えてください。

- パッケージ不足エラー
  - 該当スクリプトの import に応じて必要依存（`torch`, `pyyaml`, `Pillow` など）をインストールしてください。
