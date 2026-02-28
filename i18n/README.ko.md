[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

ILM은 텍스트를 이미지처럼 생성하는 접근을 탐구하는 연구 코드베이스입니다. 언어를 컴팩트한 이미지 유사 텐서로 인코딩하고, 확산(diffusion) 스타일의 반복적 정제를 통해 텍스트를 생성합니다. 이 표현은 문장을 메타 요소(문법, 의미, 어조, 감정)와 단어/문자를 위한 계층적 메모리형 코드로 분해합니다. 이를 통해 이산 확산, 중첩/분리(superposition/disentanglement), 구조화 임베딩, 글리프 인지 문자 모델링 아이디어를 하나로 통합합니다.

## 개요

현재 이 저장소에는 실무적으로 두 가지 큰 트랙이 포함되어 있습니다.

1. 역사적 한자 자형 어원 데이터 수집(스크래핑/파싱/저장/미리보기).
2. ILM 글리프/이미지 모델링 실험(토큰 글리프 렌더링, 프로덕트 코드북, 프레임 패킹, 확산/인페인팅, 평가/리포팅).

이 저장소의 README는 역사적으로 어원 툴킷 중심으로 작성되어 왔습니다. 해당 워크플로는 아래에 여전히 완전하게 문서화되어 있으며, 표준(canonical) 문서로 유지됩니다.

## 주요 링크

| 영역 | 경로 |
|---|---|
| 개념 설명 문서 | `docs/imagized-language-model.md` |
| 코드 계획 및 지표 | `docs/ilm-visual-diffusion-code-plan.md` |
| 임베딩 "색상" 계획 | `docs/embedding-color-plan.md` |
| 개발 노트/계획 | `docs/development-plan.md` |
| 어원 모듈 README | `ilm/etymology/README.md` |

## 기능

- 🏺 `hanziyuan`, `chineseetymology` 계열 소스에서 어원 데이터 수집.
- 🌐 재시도, 스로틀링, 캐시를 포함한 안정적인 AJAX + HTML 수집 경로.
- 🧩 `<img>` 및 CSS `background-image` data URI를 포함하는 단계 라벨 기반 글리프 추출.
- 🗃️ 문자/글리프 메타데이터와 파일시스템 자산 레이아웃을 위한 SQLite 기반 저장.
- 🖥️ 임시 수집 + 갤러리 미리보기를 위한 Tornado 웹 UI.
- 🔤 다국어 토큰 이미지용 글리프 렌더링 유틸리티.
- 🧠 프로덕트 코드 스타일 임베딩/코드북 모듈.
- 🧱 문장 프레임 패킹 및 확산/인페인팅 학습/평가 스크립트.
- 📊 임베딩 및 파이프라인 점검용 리포팅/시각화 스크립트.
- 📄 `publication/` 하위 LaTeX/PDF 출판 아티팩트.

## 프로젝트 구조

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

## 사전 요구사항

| 요구사항 | 참고 |
|---|---|
| Python `3.10+` | 핵심 런타임 |
| `pip` | 패키지 설치 |
| 선택적 GPU | PyTorch CUDA 학습 스크립트에 유용 |
| 선택적 LaTeX 툴체인 | 출판물 빌드에 필요 |

가정 메모: 현재 루트에 단일 의존성 잠금/명세 파일(`pyproject.toml`, `requirements.txt` 등)이 없어서, 의존성은 import 및 스크립트 사용 방식에서 추정합니다.

## 설치

### 최소 설치(어원 툴킷)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### 확장 설치(모델링/학습 워크플로)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

특정 스크립트에서 추가 패키지가 필요하면, 해당 스크립트 실행 시 표시되는 import 에러를 기준으로 설치하세요.

## 사용법

### 빠른 시작: 역사적 글리프 수집(CLI)

1. Hanziyuan(권장): 문자 전용 AJAX 흐름

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology(직접 URL)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. 배치 파일 수집(라인 형식: `char\turl`, `url`, `char url`)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### 출력

| 출력 유형 | 위치 |
|---|---|
| 파일 | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| 캐시 | `data/historic/cache/*.html` |
| DB | `data/historic/etymology.sqlite3` |

### 웹 데모(선택)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

`http://127.0.0.1:8888`를 열고 사이트를 선택한 뒤 문자를 입력하세요(예: `中`).

### 예의 있는 크롤링 및 사이트 존중

- 페처(fetcher)는 호스트별 스로틀링, 백오프 재시도, 캐시를 사용합니다.
- 지연 시간은 `>= 0.5s`를 유지하고, 요청 폭주를 피하며, 사이트 약관/robots/라이선스를 준수하세요.
- 유료벽(paywall)이나 인터랙티브 보호 장치를 우회하지 마세요.
- `403`/`429`가 보이면 속도를 낮추고 나중에 다시 시도하세요.

### 추가 ILM 워크플로

아래 스크립트들은 저장소의 활성 연구 워크플로에 포함되어 있지만, 로컬 데이터셋/체크포인트 준비가 필요할 수 있습니다.

1. 데이터 다운로드/준비

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. Glyph DB 라이프사이클

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. 코드/색상 모델 학습

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. 확산/인페인팅

```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_inpaint_frames.py --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt --out artifacts/inpaint
```

5. 평가/리포팅

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## 설정

주요 YAML 설정:

- `configs/color.yaml`
  - 데이터 경로: `data/processed/images_common_freq/index.tsv`
  - 모델/코드 파라미터: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - optimizer/log 설정

- `configs/diffusion.yaml`
  - 입력 JSONL: `data/processed/test_100.jsonl`
  - 프레임/그리드 + 모델 크기 설정
  - 학습 마스크 비율 범위 및 체크포인트 설정

지원되는 경우 CLI 플래그(`--epochs`, `--batch-size`, `--lr` 등)로 설정을 덮어쓸 수 있습니다.

## 예시

- 단일 영어 타일 글리프 생성:

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- 학습된 체크포인트로 인페인팅 데모 실행:

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- Hanziyuan에서 자주 쓰는 문자 대량 수집:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 개발 노트

- 이 저장소는 안정적인 CLI와 탐색적 아티팩트(노트북, 프로토타입 스크립트 포함)를 함께 담은 연구 저장소입니다.
- 생성되는 대용량 파일은 `data/` 및 `artifacts/`에 두는 것을 전제로 하며(둘 다 `.gitignore`에 포함),
- 출판물 소스와 PDF는 `publication/`에 있고, 빌드 보조 스크립트는 `scripts/latex_build.sh`입니다.
- 협업/프로세스 규약은 `AGENTS.md`에 문서화되어 있습니다.

## 문제 해결

- `ModuleNotFoundError: ilm...`
  - 저장소 루트에서 스크립트를 실행하세요.
  - 로컬 패키지 해석이 필요한 스크립트는 `PYTHONPATH=.`를 사용하세요.

- 데이터/인덱스/체크포인트에 대한 `FileNotFoundError`
  - 선행 데이터/빌드 스크립트를 먼저 실행하세요.
  - `data/processed/images_common_freq/index.tsv`, `data/processed/test_100.jsonl` 같은 기본 경로가 존재하는지 확인하세요.

- CUDA/디바이스 이슈
  - 스크립트 플래그/설정(`device: cpu` 또는 `--device cpu`)으로 CPU로 전환하세요.

- 패키지 누락 에러
  - 해당 스크립트의 import 경로에 맞춰 필요한 의존성(`torch`, `pyyaml`, `Pillow` 등)을 설치하세요.

- 스크래핑 중 HTTP `403` / `429`
  - `--delay`를 늘리고, 나중에 재시도하며, 예의 있는 요청 속도를 유지하세요.

## 로드맵

- 어원 중심 빠른 시작을 넘어 텍스트-이미지 ILM 학습/평가 런북을 계속 고도화.
- 환경 재현성 개선(단일 권위 의존성 명세).
- 연구 스크립트와 파이프라인 연결부에 대한 테스트/CI 범위 확장.
- 계층형 코드북, 확산 목표, 제어 가능 채널을 반복 개선.
- `docs/`, 스크립트 도움말, 출판 아티팩트 전반의 문서 정리/통합.

더 깊은 개념 설명과 단계별 계획은 다음 문서를 참고하세요.

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 기여

- 규약(`원자적 커밋`, `변경 후 푸시`, `코드에 자격 증명 금지`)은 `AGENTS.md`를 따르세요.
- 관련된 변경은 관례적 메시지와 함께 집중된 커밋으로 묶으세요.
- 플래그와 입력 경로를 명시한 재현 가능한 스크립트 호출을 우선하세요.
- 스크래핑 관련 변경에서는 스로틀링/캐시 동작과 사이트 존중 제약을 유지하세요.

## 라이선스

현재 이 저장소에는 최상위 라이선스 파일이 없습니다.

가정 메모: 관리자가 `LICENSE` 파일을 추가하기 전까지, 본 프로젝트는 라이선스가 명시되지 않은 연구 코드로 간주하세요.
