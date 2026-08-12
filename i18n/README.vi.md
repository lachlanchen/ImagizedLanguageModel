[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Paradigm](https://img.shields.io/badge/paradigm-retinal%20flow-16835B)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Domain](https://img.shields.io/badge/domain-historic%20etymology%20%7C%20glyph%20models-2F80ED?logo=github)

ILM là dự án nghiên cứu học và sinh ngôn ngữ dưới dạng **chữ viết nhìn thấy
được**. Mô hình hiện tại đọc các ô mực có thứ tự, duy trì trạng thái thị giác
liên tục và viết trực tiếp ô tiếp theo bằng pixel qua rectified flow. Các
codebook và mô hình khuếch tán trang cũ được giữ làm đường chuẩn so sánh, không
phải mô hình hiện hành.

## Mô hình hiện tại: Retinal Flow Language Model

![Sơ đồ Retinal Flow Language Model](../publication/ilm-image-native/figures/retinal_flow_paradigm.png)

Biên của mô hình con là `pixel chữ viết -> trạng thái võng mạc liên tục -> pixel
mực`. Mô hình không nhận token văn bản, mã Unicode, OCR, codebook thị giác hay
mô hình ngôn ngữ ngoài. Bộ đọc thị giác đạt `97,65%` về nhận dạng hình ảnh,
nhưng dự đoán với toàn bộ ngữ cảnh chỉ đạt `0,91%`, thấp hơn unigram (`1,86%`)
và bigram (`13,58%`). Sinh tự trị cũng trôi thành các nét không đọc được. Vì
vậy **MVP hiện tại bị bác bỏ với tư cách mô hình ngôn ngữ**. Bước tiếp theo là
huấn luyện trên quỹ đạo thị giác do chính mô hình tạo ra, không phải tăng kích
thước ngay lập tức.

> Kho lưu trữ duy trì cùng lúc đường ống ngữ nguyên lịch sử có tính thực dụng và các thử nghiệm ILM tầm xa trong cùng một nơi.

## 📌 Tổng quan

Repo này gồm ba hướng liên kết:

1. Mô hình ngôn ngữ ảnh Retinal Flow và đánh giá ngoài tập huấn luyện nghiêm ngặt.
2. Thu thập chữ Hán lịch sử có lưu nguồn gốc.
3. Các đường chuẩn glyph, codebook, diffusion, folio và InkStream cũ được giữ để tái lập.

README này ghi nhận cả ba hướng và giữ luồng etymology như một phần cốt lõi có thể lặp lại.

## 🔗 Liên kết chính

| Khu vực | Đường dẫn |
|---|---|
| Bài viết khái niệm | `docs/imagized-language-model.md` |
| Mục tiêu kỹ thuật hiện tại | `docs/first-imagized-language-model-goal.md` |
| Hồ sơ nghiên cứu và bằng chứng | `references/image-native-language-model-research.md` |
| Kế hoạch mã nguồn và chỉ số | `docs/ilm-visual-diffusion-code-plan.md` |
| Kế hoạch "màu" embedding | `docs/embedding-color-plan.md` |
| Ghi chú/kế hoạch phát triển | `docs/development-plan.md` |
| README module Etymology | `ilm/etymology/README.md` |

## ✨ Tính năng

- 🏺 Thu thập etymology từ các nguồn kiểu `hanziyuan` và `chineseetymology`.
- 🌐 Luồng thu thập AJAX + HTML ổn định với retry, throttle và cache.
- 🧩 Trích xuất glyph có gắn nhãn theo giai đoạn, gồm dữ liệu URI của `<img>` và `background-image` trong CSS.
- 🗃️ Lưu trữ bằng SQLite cho metadata ký tự/glyph kèm bố cục tài nguyên trên filesystem.
- 🖥️ Giao diện web Tornado cho ingest theo yêu cầu và xem trước gallery.
- 🔤 Công cụ render glyph cho token đa ngôn ngữ.
- 🧠 Module embedding/codebook theo phong cách product-code.
- 🧱 Đóng gói frame câu và scripts huấn luyện/đánh giá khuếch tán/phục hồi.
- 📊 Scripts báo cáo và trực quan hóa cho kiểm tra embedding và pipeline.
- 📄 Tài liệu phát hành bằng LaTeX/PDF trong `publication/`.

## 🧱 Cấu trúc dự án

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

## 🧰 Điều kiện tiên quyết

| Yêu cầu | Ghi chú |
|---|---|
| Python `3.10+` | Môi trường chạy chính |
| `pip` | Cài đặt gói |
| GPU tùy chọn | Có ích cho các script huấn luyện PyTorch CUDA |
| Chuỗi công cụ LaTeX tùy chọn | Cần cho build publication |

Lưu ý giả định: hiện chưa có một file khóa phụ thuộc gốc tại root (`pyproject.toml`, `requirements.txt`, ...), nên dependency được suy ra từ `import` và cách dùng script.

## ⚙️ Cài đặt

### Tối thiểu (bộ công cụ etymology)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### Mở rộng (quy trình mô hình hóa/huấn luyện)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

Nếu một script cụ thể cần thêm gói, hãy cài từ lỗi import mà script đó báo ra.

## 🚀 Cách dùng

### Khởi động nhanh: Thu thập glyph lịch sử (CLI)

1. Hanziyuan (khuyến nghị): luồng AJAX chỉ theo ký tự

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (URL trực tiếp)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. Nhập hàng loạt từ file (mỗi dòng có thể là `char\turl`, `url`, hoặc `char url`)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### Kết quả đầu ra

| Loại đầu ra | Vị trí |
|---|---|
| Tệp tin | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| Cache | `data/historic/cache/*.html` |
| Cơ sở dữ liệu | `data/historic/etymology.sqlite3` |

### Demo web (tùy chọn)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

Mở `http://127.0.0.1:8888`, chọn site, nhập ký tự (ví dụ `中`).

### Crawl lịch sự và tôn trọng site

- Bộ lấy dữ liệu dùng throttle theo host, retry có backoff và cache.
- Giữ độ trễ `>= 0.5s`, tránh burst và tuân thủ điều khoản/robots/licensing của site.
- Không vượt rào cản bản quyền hay các cơ chế bảo vệ tương tác.
- Nếu thấy `403`/`429`, hãy giảm tốc và thử lại sau.

### Luồng làm việc ILM bổ sung

Các script này hiện có trong repo và đang hoạt động trong thực tiễn nghiên cứu, nhưng thường đòi hỏi dữ liệu/checkpoint địa phương đã chuẩn bị sẵn.

1. Tải/xử lý dữ liệu

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. Chu trình Glyph DB

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. Huấn luyện mã/màu

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. Khuếch tán/phục hồi một phần

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

## 🧩 Cấu hình

Các cấu hình YAML chính:

- `configs/color.yaml`
  - Đường dẫn dữ liệu: `data/processed/images_common_freq/index.tsv`
  - Tham số model/mã: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - Cài đặt optimizer/log

- `configs/diffusion.yaml`
  - JSONL đầu vào: `data/processed/test_100.jsonl`
  - Khung lưới + cài đặt kích thước model
  - Khoảng tỉ lệ che mask trong huấn luyện và cài đặt checkpoint

Ghi đè cài đặt qua CLI flags nếu hỗ trợ (`--epochs`, `--batch-size`, `--lr`, ...).

## 🧪 Ví dụ

- Tạo glyph tile tiếng Anh đơn lẻ:

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

- Nhập hàng loạt ký tự phổ biến từ Hanziyuan:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 📝 Ghi chú phát triển

- Đây là repo nghiên cứu với cả CLI ổn định và các artifact thăm dò (kể cả notebook và script nguyên mẫu).
- Các file lớn được đặt trong `data/` và `artifacts/` (cả hai đều bị bỏ qua trong `.gitignore`).
- Tài liệu và PDF phát hành nằm trong `publication/`; script hỗ trợ build: `scripts/latex_build.sh`.
- Các quy ước hợp tác/quy trình được ghi trong `AGENTS.md`.

## 🛠️ Khắc phục sự cố

- `ModuleNotFoundError: ilm...`
  - Chạy scripts từ thư mục gốc repository.
  - Dùng `PYTHONPATH=.` cho các script đòi hỏi giải quyết gói nội bộ.

- `FileNotFoundError` cho dữ liệu/index/checkpoint
  - Chạy trước các script chuẩn bị dữ liệu/build.
  - Kiểm tra chắc chắn các mặc định như `data/processed/images_common_freq/index.tsv` và `data/processed/test_100.jsonl` tồn tại.

- Vấn đề CUDA/device
  - Chuyển sang CPU bằng cờ cấu hình/script (`device: cpu` hoặc `--device cpu`).

- Lỗi thiếu gói phụ thuộc
  - Cài dependency cần thiết theo đường dẫn import của script (`torch`, `pyyaml`, `Pillow`, ...).

- HTTP `403` / `429` khi crawl
  - Tăng `--delay`, retry lại sau và giữ yêu cầu lịch sự.

## 🗺️ Lộ trình

- Tiếp tục làm giàu quy trình chạy thử nghiệm huấn luyện/đánh giá ILM text-as-image ngoài luồng khởi tạo nhanh etymology.
- Cải thiện khả năng tái lập môi trường (một file khóa dependency chuẩn duy nhất).
- Mở rộng test/CI cho script nghiên cứu và phần ghép pipeline.
- Lặp lại về codebook phân cấp, hàm mục tiêu khuếch tán và kênh controllability.
- Hợp nhất tài liệu giữa `docs/`, help text trong script và các artifact publication.

Để biết chi tiết kế hoạch khái niệm theo giai đoạn sâu hơn, xem:

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 🤝 Đóng góp

- Tuân thủ `AGENTS.md` cho các quy ước (commit nguyên tử, đẩy sau mỗi lần thay đổi, không để credential trong code).
- Gom các thay đổi liên quan vào commit tập trung với thông điệp theo chuẩn.
- Ưu tiên gọi script tái lập có cờ rõ ràng và đường dẫn đầu vào xác định.
- Với thay đổi liên quan đến scraping, giữ nguyên hành vi throttle/cache và tôn trọng điều kiện site.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

Không có tệp giấy phép cấp cao ở root repository hiện tại.

Lưu ý giả định: hãy coi dự án như mã nghiên cứu với giấy phép chưa xác định cho đến khi maintainers thêm file `LICENSE`.
