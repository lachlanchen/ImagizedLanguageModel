Imagized Language Model (ILM)
=============================

言語
- [English](../README.md) | [简体中文](README.zh-Hans.md) | [繁體中文](README.zh-Hant.md) | 日本語 | [한국어](README.ko.md) | [Tiếng Việt](README.vi.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Español](README.es.md)

概要
ILM はテキストを「画像のような」テンソルにエンコードし、拡散的な逐次デノイズでテキストを生成します。文を、制御可能なメタ要素（文法・意味・トーン・感情）と、階層的な「メモリ風」コード（単語/字形）に分解します。離散拡散、重ね合わせ/分離、構造化埋め込み、グリフ認識を統合します。

主要リンク
- コンセプト: docs/imagized-language-model.md
- 設計/指標: docs/ilm-visual-diffusion-code-plan.md
- 埋め込み「色」計画: docs/embedding-color-plan.md

内容
- ilm/etymology/: 甲骨/金文/篆書など歴史的字形の収集ツール
  - hanziyuan への AJAX 取得（リトライ、スロットリング、キャッシュ）
  - HTML/CSS を解析して段階ラベル付き画像を抽出（data URI / URL）
- scripts/
  - ingest_etymology.py: CLI で取得し SQLite/ファイルに保存
  - serve_etymology.py: Tornado ベースの簡易 UI
  - use_historic_tools.md: 外部データ/ツールのメモ
- data/（git ignore）: HTML キャッシュ、画像、SQLite DB

クイックスタート
- 依存: `pip install requests beautifulsoup4 tornado`
- 例（hanziyuan AJAX 推奨）:
  - `PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中`
- Web デモ:
  - `PYTHONPATH=. python scripts/serve_etymology.py` → http://127.0.0.1:8888

出力先
- 画像: data/historic/glyphs/<字>/<段階>/<ラベル>.<拡張子>
- キャッシュ: data/historic/cache
- DB: data/historic/etymology.sqlite3

丁寧な取得
- ホスト単位のスロットリング、リトライ/バックオフ、キャッシュを実装
- サイト規約/ライセンスを尊重。403/429 時は速度を下げて再試行

目的
- 一般的な PC で学習/推論できる、構造化/制御可能/多言語グリフ対応のモデル

貢献
- AGENTS.md に従ってください（原子的コミット、毎回プッシュ、資格情報をコミットしない）

