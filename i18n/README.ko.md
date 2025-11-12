Imagized Language Model (ILM)
=============================

언어
- [English](../README.md) | [简体中文](README.zh-Hans.md) | [繁體中文](README.zh-Hant.md) | [日本語](README.ja.md) | 한국어 | [Tiếng Việt](README.vi.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Español](README.es.md)

개요
ILM은 텍스트를 ‘이미지 같은’ 텐서로 표현하고, 확산(디퓨전) 방식의 단계적 디노이즈로 문장을 생성합니다. 문장을 메타 요소(문법/의미/톤/감정)와 계층적 ‘메모리형’ 코드(단어/글리프)로 분해합니다. 이는 이산 확산, 특성 중첩/분리, 구조적 임베딩, 글리프 인식 아이디어를 통합합니다.

핵심 링크
- 개념 문서: docs/imagized-language-model.md
- 코드 계획/지표: docs/ilm-visual-diffusion-code-plan.md
- 임베딩 ‘색’ 계획: docs/embedding-color-plan.md

구성
- ilm/etymology/: 한자 어원 글리프(갑골/금문/전서 등) 수집 도구
  - hanziyuan AJAX 수집(재시도/스로틀/캐시)
  - HTML/CSS 파싱으로 단계 라벨 이미지 추출(data URI/URL)
- scripts/
  - ingest_etymology.py: CLI 수집 → SQLite/파일 저장
  - serve_etymology.py: Tornado 기반 미니 UI
  - use_historic_tools.md: 외부 데이터/도구 사용법
- data/(git ignore): HTML 캐시, 이미지, SQLite DB

빠른 시작
- 의존성: `pip install requests beautifulsoup4 tornado`
- 예시(hanziyuan AJAX 권장):
  - `PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中`
- 웹 데모:
  - `PYTHONPATH=. python scripts/serve_etymology.py` → http://127.0.0.1:8888

출력 경로
- 이미지: data/historic/glyphs/<한자>/<단계>/<라벨>.<확장자>
- 캐시: data/historic/cache
- DB: data/historic/etymology.sqlite3

예의 바른 수집
- 호스트 단위 스로틀/백오프 재시도/캐시 적용
- 사이트 약관/라이선스 준수. 403/429 시 속도를 낮추고 재시도

프로젝트 목표
- 일반 컴퓨터에서도 학습/추론 가능한 구조적·제어가능·다국어 글리프 지원 모델

기여
- AGENTS.md 준수(원자적 커밋, 변경 후 즉시 푸시, 자격 증명 불커밋)

