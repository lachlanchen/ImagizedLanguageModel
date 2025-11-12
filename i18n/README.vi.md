Imagized Language Model (ILM)
=============================

Ngôn ngữ
- [English](../README.md) | [简体中文](README.zh-Hans.md) | [繁體中文](README.zh-Hant.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | Tiếng Việt | [العربية](README.ar.md) | [Français](README.fr.md) | [Español](README.es.md)

Tổng quan
ILM mã hoá văn bản thành tensor “giống ảnh” gọn nhẹ và tạo văn bản bằng khử nhiễu lặp kiểu diffusion. Câu được tách thành các siêu‑yếu tố có thể điều khiển (ngữ pháp, ngữ nghĩa, giọng điệu, cảm xúc) và mã phân cấp kiểu “bộ nhớ” (từ/ký tự). Kết hợp diffusion rời rạc, chồng chất/giải ghép đặc trưng, nhúng có cấu trúc và nhận biết glyph.

Liên kết chính
- Bài viết khái niệm: docs/imagized-language-model.md
- Kế hoạch mã & chỉ số: docs/ilm-visual-diffusion-code-plan.md
- Kế hoạch “màu” nhúng: docs/embedding-color-plan.md

Thành phần
- ilm/etymology/: công cụ thu thập glyph Hán cổ (giáp cốt/kim văn/triện thư…)
  - Lấy AJAX từ hanziyuan (retry, throttle, cache)
  - Phân tích HTML/CSS và trích ảnh có nhãn giai đoạn (data URI/URL)
- scripts/
  - ingest_etymology.py: CLI thu thập → lưu SQLite & tệp
  - serve_etymology.py: UI Tornado nhỏ để xem nhanh
  - use_historic_tools.md: ghi chú dữ liệu/công cụ ngoài
- data/ (bỏ qua git): cache HTML, ảnh, CSDL SQLite

Bắt đầu nhanh
- Phụ thuộc: `pip install requests beautifulsoup4 tornado`
- Ví dụ (khuyên dùng hanziyuan AJAX):
  - `PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中`
- Web demo:
  - `PYTHONPATH=. python scripts/serve_etymology.py` → http://127.0.0.1:8888

Đầu ra
- Ảnh: data/historic/glyphs/<chữ>/<giai‑đoạn>/<nhãn>.<đuôi>
- Cache: data/historic/cache
- CSDL: data/historic/etymology.sqlite3

Thu thập lịch sự
- Throttle theo host, retry/backoff, và cache đã được bật
- Tôn trọng điều khoản/giấy phép. Nếu 403/429, giảm tốc và thử lại sau

Mục tiêu
- Mô hình có cấu trúc, điều khiển được, hỗ trợ đa ngôn ngữ, chạy được trên máy phổ thông

Đóng góp
- Theo AGENTS.md (commit nguyên tử, push sau mỗi thay đổi, không commit thông tin nhạy cảm)

