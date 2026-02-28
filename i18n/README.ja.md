[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Domain](https://img.shields.io/badge/domain-historic%20etymology%20%7C%20glyph%20models-2F80ED?logo=github)

ILM は **text-as-image generation** を扱う研究用コードベースです。言語をコンパクトな画像ライクなテンソルとして符号化し、拡散スタイルの反復的な精緻化でテキストを生成します。表現は文をメタ要素（文法、意味、トーン、感情）と、単語・文字向けの階層的なメモリ的コードに分解します。これにより、離散拡散、重ね合わせ/独立化、構造化埋め込み、字形を意識した文字モデリングという発想を統合します。

> リポジトリは実用的な語源パイプラインと長期の ILM 実験を意図的に並行して維持しています。

## 📌 概要

このリポジトリには以下の2つのアクティブなトラックがあります。

1. 歴史的漢字字形の語源データ取り込み（スクレイピング / パース / 保存 / プレビュー）。
2. ILM のグリフ/画像モデリング実験（トークングリフ描画、コードブック、フレームパッキング、拡散/インペインティング、評価/レポート）。

この README は両トラックを同時に記載し、語源ワークフローを本流で再現可能な主要経路として扱います。

## 🔗 主要リンク

| 領域 | パス |
|---|---|
| 概念説明 | `docs/imagized-language-model.md` |
| コード計画と指標 | `docs/ilm-visual-diffusion-code-plan.md` |
| 埋め込み「カラー」計画 | `docs/embedding-color-plan.md` |
| 開発ノート/計画 | `docs/development-plan.md` |
| 語源モジュール README | `ilm/etymology/README.md` |

## ✨ 特徴

- 🏺 `hanziyuan` と `chineseetymology` 形式の語源データ取り込み。
- 🌐 リトライ、間引き（throttling）、キャッシュを備えた堅牢な AJAX + HTML 取り込みフロー。
- 🧩 ステージラベル付きのグリフ抽出で、`<img>` と CSS `background-image` の data URI を扱う。
- 🗃️ 文字／グリフメタデータとファイルシステム資産レイアウトを管理する SQLite バックエンド。
- 🖥️ アドホック取り込みとギャラリープレビュー向け Tornado Web UI。
- 🔤 多言語トークン画像向けのグリフ描画ユーティリティ。
- 🧠 製品コード（Product-code）風の埋め込み/コードブックモジュール。
- 🧱 文フレームのパッキングと拡散／インペインティングの学習・評価スクリプト。
- 📊 埋め込みとパイプライン検査用のレポート・可視化スクリプト。
- 📄 `publication/` 配下の LaTeX/PDF 形式の公開用成果物。

## 🧱 プロジェクト構成

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

## 🧰 前提条件

| 要件 | 補足 |
|---|---|
| Python `3.10+` | コア実行環境 |
| `pip` | パッケージ導入 |
| GPU（任意） | PyTorch CUDA 学習スクリプトの実行に有効 |
| LaTeX ツールチェーン（任意） | 公開物のビルドに必要 |

前提メモ: 現時点で、`pyproject.toml` や `requirements.txt` のような単一のルート依存ファイルはありません。依存関係は import とスクリプト利用例から推定します。

## ⚙️ インストール

### 最小構成（語源ツールキット）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### 拡張構成（モデリング／学習ワークフロー）

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

特定スクリプトで追加パッケージが必要な場合は、そのスクリプトの import エラー表示に従って個別に導入してください。

## 🚀 使用方法

### クイックスタート: 歴史的グリフ取り込み（CLI）

1. Hanziyuan（推奨）: 文字のみの AJAX フロー

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology（URL を直接指定）

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. バッチファイル取り込み（各行は `char\turl`、`url`、`char url` のいずれか）

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### 出力

| 出力タイプ | 保存場所 |
|---|---|
| ファイル | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| キャッシュ | `data/historic/cache/*.html` |
| DB | `data/historic/etymology.sqlite3` |

### Web デモ（任意）

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

`http://127.0.0.1:8888` を開き、サイトを選択して文字（例: `中`）を入力します。

### スクレイピングの礼儀とサイトの尊重

- フェッチャーはホストごとの間引き、指数バックオフ付きリトライ、キャッシュを使用します。
- 遅延は `>= 0.5s` を維持し、短時間の連続リクエストを避け、サイトの利用規約・robots・ライセンスを尊重してください。
- ペイウォールや対話式の保護を回避してはいけません。
- `403` / `429` が発生した場合は、リトライ間隔を延ばしてしばらく待ってから再実行してください。

### 追加の ILM ワークフロー

これらのスクリプトは現在リポジトリ上で公開されており、研究向けのワークフローです。ローカルで準備済みのデータセットやチェックポイントが必要になる場合があります。

1. データ取得 / 前処理

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. Glyph DB のライフサイクル

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. コード/カラー モデルの学習

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. 拡散／インペインティング

```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_inpaint_frames.py --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt --out artifacts/inpaint
```

5. 評価／レポート

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## 🧩 設定

主要な YAML 設定:

- `configs/color.yaml`
  - データパス: `data/processed/images_common_freq/index.tsv`
  - モデル・コードパラメータ: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - optimizer / ログ設定

- `configs/diffusion.yaml`
  - 入力 JSONL: `data/processed/test_100.jsonl`
  - フレーム・グリッドとモデルサイズの設定
  - 学習時マスク率の範囲およびチェックポイント設定

対応する場合は CLI フラグ（`--epochs`, `--batch-size`, `--lr` など）で設定を上書きできます。

## 🧪 例

- 英語 1 文字タイルのグリフを作成:

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- 学習済みチェックポイントを使ってインペインティングデモを実行:

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- Hanziyuan から高頻度文字を一括取り込み:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 📝 開発ノート

- このリポジトリは、堅牢な CLI と探索的アーティファクト（ノートブックや試作スクリプトを含む）を併せ持つ研究用コードベースです。
- 大きな生成ファイルは `data/` と `artifacts/` に置く想定です（どちらも `.gitignore` で除外）。
- 公開用ソースと PDF は `publication/` 配下にあり、補助スクリプトは `scripts/latex_build.sh`。
- 協業・進行ルールは `AGENTS.md` に記載されています。

## 🛠️ トラブルシューティング

- `ModuleNotFoundError: ilm...`
  - リポジトリのルートからスクリプトを実行してください。
  - ローカルパッケージ解決を期待するスクリプトでは `PYTHONPATH=.` を使います。

- `FileNotFoundError`（data / index / チェックポイント）
  - 先に前提となるデータ作成スクリプトを実行してください。
  - `data/processed/images_common_freq/index.tsv` や `data/processed/test_100.jsonl` が存在することを確認してください。

- CUDA / デバイスに関する問題
  - スクリプトのフラグまたは設定（`device: cpu` または `--device cpu`）で CPU 実行に切り替えてください。

- パッケージ不足エラー
  - 該当スクリプトの import が示す不足パッケージ（`torch`、`pyyaml`、`Pillow` など）をインストールしてください。

- スクレイピング時の HTTP `403` / `429`
  - `--delay` を増やし、時間を空けて再試行してください。

## 🗺️ ロードマップ

- まずは歴史的語源の高速スタートを越えて、text-as-image ILM の学習・評価ランブックを成熟させる。
- 環境の再現性を改善（単一の権威ある依存定義の整備）。
- 研究スクリプトとパイプライン結合部のテスト/CI を拡充する。
- 階層的コードブック、拡散目的関数、制御性チャネルを継続的に改善する。
- `docs/`、スクリプトのヘルプ、公開成果物でドキュメントを統合する。

より深いコンセプトと段階別計画については、次を参照してください。

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 🤝 コントリビュート

- `AGENTS.md` の規約（原則コミット、変更後の push、コードへの認証情報埋め込みなし）を守ってください。
- 関連作業は論理的に関連する変更を1コミットにまとめます。
- 再現可能性の高いスクリプト実行（明示的フラグと入力パス指定）を優先します。
- スクレイピングに関する変更は、レート制御・キャッシュ挙動・サイト尊重の制約を維持します。

## 📄 ライセンス

トップレベルのライセンスファイルは現在このリポジトリにありません。

前提メモ: `LICENSE` ファイルが追加されるまで、保守者側で未確定のライセンスとして扱ってください。


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
