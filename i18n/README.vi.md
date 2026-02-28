[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

ILM là một codebase nghiên cứu khám phá hướng sinh văn bản dưới dạng ảnh (text-as-image): hệ thống mã hóa ngôn ngữ thành các tensor cô đọng giống ảnh và sinh văn bản bằng tinh chỉnh lặp theo kiểu diffusion. Biểu diễn này phân rã câu thành các siêu thành phần (ngữ pháp, ngữ nghĩa, giọng điệu, cảm xúc) cùng các mã phân cấp kiểu bộ nhớ cho từ và ký tự. Cách tiếp cận này hợp nhất các ý tưởng từ discrete diffusion, superposition/disentanglement, embedding có cấu trúc và mô hình hóa ký tự có nhận thức glyph.

## Tổng quan

Kho mã hiện bao gồm hai hướng thực nghiệm chính:

1. Thu nạp từ nguyên chữ Hán cổ (scraping/parsing/lưu trữ/xem trước).
2. Thử nghiệm mô hình hóa glyph/ảnh của ILM (render glyph token, product codebook, đóng gói frame, diffusion/inpainting, đánh giá/báo cáo).

README hiện tại của repo này trước đây chủ yếu tập trung vào bộ công cụ từ nguyên. Quy trình đó vẫn được tài liệu hóa đầy đủ bên dưới và được giữ làm chuẩn (canonical).

## Liên kết chính

| Khu vực | Đường dẫn |
|---|---|
| Tài liệu khái niệm | `docs/imagized-language-model.md` |
| Kế hoạch mã và chỉ số | `docs/ilm-visual-diffusion-code-plan.md` |
| Kế hoạch “màu” embedding | `docs/embedding-color-plan.md` |
| Ghi chú/kế hoạch phát triển | `docs/development-plan.md` |
| README mô-đun từ nguyên | `ilm/etymology/README.md` |

## Tính năng

- 🏺 Thu nạp dữ liệu từ nguyên từ các nguồn kiểu `hanziyuan` và `chineseetymology`.
- 🌐 Luồng thu nạp AJAX + HTML ổn định với retry, throttling và cache.
- 🧩 Trích xuất glyph có nhãn theo giai đoạn, gồm cả `<img>` và CSS `background-image` dạng data URI.
- 🗃️ Lưu trữ dựa trên SQLite cho metadata ký tự/glyph cùng bố cục tài nguyên trên filesystem.
- 🖥️ Giao diện web Tornado để thu nạp ad-hoc và xem trước gallery.
- 🔤 Tiện ích render glyph cho ảnh token đa ngôn ngữ.
- 🧠 Các mô-đun embedding/codebook theo phong cách product-code.
- 🧱 Script huấn luyện/đánh giá đóng gói frame câu và diffusion/inpainting.
- 📊 Script báo cáo và trực quan hóa cho embedding và kiểm tra pipeline.
- 📄 Tạo phẩm xuất bản LaTeX/PDF trong `publication/`.

## Cấu trúc dự án

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

## Điều kiện tiên quyết

| Yêu cầu | Ghi chú |
|---|---|
| Python `3.10+` | Môi trường chạy chính |
| `pip` | Cài đặt package |
| GPU tùy chọn | Hữu ích cho script huấn luyện PyTorch CUDA |
| Bộ công cụ LaTeX tùy chọn | Cần cho build tài liệu xuất bản |

Ghi chú giả định: hiện chưa có một tệp khóa/đặc tả phụ thuộc ở thư mục gốc (`pyproject.toml`, `requirements.txt`, v.v.), nên phụ thuộc được suy ra từ import và cách dùng script.

## Cài đặt

### Tối thiểu (bộ công cụ từ nguyên)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### Mở rộng (workflow mô hình hóa/huấn luyện)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

Nếu một script cụ thể cần thêm package, hãy cài theo lỗi import mà script đó báo.

## Cách dùng

### Bắt đầu nhanh: Thu nạp Glyph Lịch sử (CLI)

1. Hanziyuan (khuyến nghị): luồng AJAX chỉ với ký tự

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (URL trực tiếp)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. Thu nạp từ tệp batch (mỗi dòng có thể là `char\turl`, `url`, hoặc `char url`)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### Đầu ra

| Loại đầu ra | Vị trí |
|---|---|
| Tệp | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| Cache | `data/historic/cache/*.html` |
| DB | `data/historic/etymology.sqlite3` |

### Web Demo (tùy chọn)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

Mở `http://127.0.0.1:8888`, chọn site, nhập một ký tự (ví dụ `中`).

### Thu thập lịch sự và tôn trọng website

- Trình thu thập dùng throttling theo host, retry với backoff và cache.
- Giữ độ trễ `>= 0.5s`, tránh gửi dồn dập, và tôn trọng điều khoản/robots/giấy phép của website.
- Không vượt paywall hoặc cơ chế bảo vệ tương tác.
- Nếu gặp `403`/`429`, hãy giảm tốc và thử lại sau.

### Các workflow ILM bổ sung

Các script này có trong repo và là một phần tích cực của bề mặt nghiên cứu, nhưng là workflow nghiên cứu nên có thể cần dataset/checkpoint đã chuẩn bị sẵn trên máy cục bộ.

1. Tải dữ liệu/tiền xử lý

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. Vòng đời Glyph DB

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. Huấn luyện mô hình code/màu

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

5. Đánh giá/báo cáo

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## Cấu hình

Các tệp cấu hình YAML chính:

- `configs/color.yaml`
  - đường dẫn dữ liệu: `data/processed/images_common_freq/index.tsv`
  - tham số model/code: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - thiết lập optimizer/log

- `configs/diffusion.yaml`
  - JSONL đầu vào: `data/processed/test_100.jsonl`
  - thiết lập kích thước frame/grid + model
  - khoảng tỉ lệ mask khi train và thiết lập checkpoint

Ghi đè cấu hình qua cờ CLI ở những script có hỗ trợ (`--epochs`, `--batch-size`, `--lr`, v.v.).

## Ví dụ

- Tạo một tile glyph tiếng Anh đơn lẻ:

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- Chạy demo inpainting với checkpoint đã huấn luyện:

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- Thu nạp hàng loạt các ký tự phổ biến từ Hanziyuan:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## Ghi chú phát triển

- Đây là repo nghiên cứu với cả CLI tương đối hoàn chỉnh và các hiện vật khám phá (bao gồm notebook và script nguyên mẫu).
- Các tệp lớn được sinh ra dự kiến nằm trong `data/` và `artifacts/` (đều đã nằm trong `.gitignore`).
- Mã nguồn và PDF xuất bản nằm trong `publication/`; script build hỗ trợ: `scripts/latex_build.sh`.
- Quy ước cộng tác/quy trình được mô tả trong `AGENTS.md`.

## Khắc phục sự cố

- `ModuleNotFoundError: ilm...`
  - Chạy script từ thư mục gốc của repo.
  - Dùng `PYTHONPATH=.` cho các script cần phân giải package cục bộ.

- `FileNotFoundError` cho data/index/checkpoint
  - Chạy trước các script dữ liệu/build bắt buộc.
  - Xác nhận các đường dẫn mặc định như `data/processed/images_common_freq/index.tsv` và `data/processed/test_100.jsonl` có tồn tại.

- Lỗi CUDA/device
  - Chuyển sang CPU bằng cờ/script config (`device: cpu` hoặc `--device cpu`).

- Lỗi thiếu package
  - Cài phụ thuộc cần thiết theo import của script cụ thể (`torch`, `pyyaml`, `Pillow`, v.v.).
