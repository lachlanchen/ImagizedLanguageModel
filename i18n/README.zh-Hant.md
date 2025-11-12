ILM（影像化語言模型）
=====================

語言
- [English](../README.md) | [简体中文](README.zh-Hans.md) | 繁體中文 | [日本語](README.ja.md) | [한국어](README.ko.md) | [Tiếng Việt](README.vi.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Español](README.es.md)

簡介
ILM 將「文字」表示為緊湊的「影像式」張量，並以擴散式逐步去噪產生文本。表示把句子分解為可控的元要素（語法、語義、語氣、情感）以及分層的「記憶式」編碼（單詞/字形）。融合離散擴散、特徵疊加/解耦、結構化嵌入與字形感知建模。

關鍵連結
- 概念說明：docs/imagized-language-model.md
- 程式計畫與指標：docs/ilm-visual-diffusion-code-plan.md
- 嵌入「顏色」計畫：docs/embedding-color-plan.md

倉庫內容
- ilm/etymology/：蒐集中文字源字形（甲骨/金文/篆書等）工具
  - hanziyuan AJAX 抓取（重試、節流、快取）
  - 解析 HTML/CSS，擷取分階段圖像（data URI / URL）
- scripts/
  - ingest_etymology.py：CLI 抓取並寫入 SQLite 與檔案
  - serve_etymology.py：簡易 Tornado UI，示範擷取與預覽
  - use_historic_tools.md：外部資料/工具說明
- data/（已忽略）：HTML 快取、字形、SQLite 資料庫

快速開始
1) 環境：Python 3.10+
2) 安裝依賴：`pip install requests beautifulsoup4 tornado`
3) 抓取範例（建議 hanziyuan AJAX）：
- `PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中`
4) Web 展示：
- `PYTHONPATH=. python scripts/serve_etymology.py`，開啟 http://127.0.0.1:8888

輸出位置
- 影像：data/historic/glyphs/<字>/<階段>/<標籤>.<副檔名>
- 快取：data/historic/cache
- 資料庫：data/historic/etymology.sqlite3

禮貌抓取
- 內建主機級節流、重試/回退與快取；建議延遲 ≥ 0.5s
- 尊重站點條款/授權；遇 403/429 請降低速率稍後再試

專案目標
- 在一般電腦上可訓練/推理的文本‑影像整合方案；強調結構化、可控性與多語字形能力

貢獻規範
- 參見 AGENTS.md（原子提交、每次修改後推送、勿提交憑證）

