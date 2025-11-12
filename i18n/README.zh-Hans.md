ILM（图像化语言模型）
=====================

语言
- [English](../README.md) | 简体中文 | [繁體中文](README.zh-Hant.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Tiếng Việt](README.vi.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Español](README.es.md)

简介
ILM 将“文本”表示为紧凑的“图像样”张量，并使用扩散式的逐步去噪来生成文本。表示把句子分解为可控的元要素（语法、语义、语气、情感）以及分层的“记忆式”编码（单词/字形）。该思路融合了离散扩散、特征叠加/解耦、结构化嵌入与字形感知建模。

关键链接
- 概念说明：docs/imagized-language-model.md
- 代码计划与指标：docs/ilm-visual-diffusion-code-plan.md
- 嵌入“颜色”计划：docs/embedding-color-plan.md

仓库内容
- ilm/etymology/：收集汉字字源字形（甲骨/金文/篆书等）的工具
  - 针对 hanziyuan 的 AJAX 抓取（带重试、节流与缓存）
  - 解析 HTML/CSS，抽取带阶段标签的图像（data URI / URL）
- scripts/
  - ingest_etymology.py：命令行抓取并写入 SQLite 与本地文件
  - serve_etymology.py：简易 Tornado UI，演示采集与预览
  - use_historic_tools.md：外部数据/工具的使用说明
- data/（已忽略）：缓存 HTML、字形文件、SQLite 数据库

快速开始
1) 环境：Python 3.10+
2) 安装依赖：`pip install requests beautifulsoup4 tornado`
3) 抓取示例（推荐 hanziyuan AJAX）：
- `PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中`
4) Web 演示：
- `PYTHONPATH=. python scripts/serve_etymology.py`，打开 http://127.0.0.1:8888

输出位置
- 图像：data/historic/glyphs/<字>/<阶段>/<标签>.<扩展名>
- 缓存：data/historic/cache
- 数据库：data/historic/etymology.sqlite3

礼貌抓取
- 脚本内置主机级节流、重试/回退与缓存；建议延时 ≥ 0.5s
- 尊重站点条款/许可；遇到 403/429 请降低速率稍后再试

项目目标
- 在普通电脑上可训练/推理的文本‑图像一体化方案；强调结构化、可控性与多语言字形能力

贡献规范
- 参见 AGENTS.md（原子提交、每次修改后推送、勿提交凭据）

