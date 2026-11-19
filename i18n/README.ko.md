[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

**왜 사람을 잘못 고를까?** 중국어로 누군지 묻는 질문 387개에서 전체 지문을 읽힌 모델은 62개를 맞혔다. 질문과 글자 모양이 겹치는 문장을 먼저 고르자 116개를 맞혔다. 추가 학습의 정답을 무작위로 배정한 대조 모델은 같은 방법으로 127개를 맞혔다. 그림 85는 발명가의 이름을 물었을 때 지명을 쓴 사례다. [결과와 실행 방법](../references/inverse_role_writer_result.md)

[![그림 85. 문장 선택과 실제 출력](../publication/ilm-nature/figures/Figure_85_inverse_role_writer.png)](../publication/ilm-nature/manuscript.pdf#page=113)

**누구에 관한 문장인가.** 새 이름과 보어로 만든 중국어 문장 404개 중 368개를 정확히 썼다. 진과 초의 금문 글자를 입력에서 옮겨 쓴 문장은 32개 중 24개가 맞았다. 이름 없이 조건만 주고 누구인지 물으면 금문 이름을 정확히 쓴 경우는 없었다. [결과 및 실행 방법](../references/visual_role_writer_result.md)

[![그림 84. 금문이 들어간 두 지문과 실제 출력](../publication/ilm-nature/figures/Figure_84_visual_role_writer.png)](../publication/ilm-nature/manuscript.pdf#page=112)

**모델은 글자 이미지에서 무엇을 읽을 수 있을까?** 같은 문장 이미지 “她喝了一口□。”(그녀는 □를 한 모금 마셨다)에서 앞 문단의 음료가 물에서 차로 바뀌자 모델의 답도 水에서 茶로 바뀌었다. 신경망과 이웃 글자의 빈도를 결합해 중국어 빈칸 2,947개 중 651개를 원문과 똑같이 복원했다. 두 인물의 역할을 맞바꾼 48쌍에서는 두 경우를 모두 올바르게 판단한 쌍이 없었다. [실험 결과와 실행 명령](../references/visual_inference_writer_result.md)

[![그림 83. 두 지문, 같은 빈칸 문장과 두 답안 이미지](../publication/ilm-nature/figures/Figure_83_visual_inference_writer.png)](../publication/ilm-nature/manuscript.pdf#page=111)

<!-- paired-writing-status:start -->
512회 업데이트한 여백 제거 비교에서도 안정적인 베껴 쓰기는 달성하지 못했습니다. OCR 완전 일치는 훈련 6/32, 검증 10/32이며, 깨끗한 참조 이미지는 27/32, 28/32입니다. 원본 출력에는 글자 누락과 반복이 보입니다. 훈련 시간은 795.624초에서 287.182초로 줄었지만 언어 이해와 실용적인 품질에서의 효율성은 입증되지 않았습니다. [64개 원본 출력](../references/paired_padding_result.md).
<!-- paired-writing-status:end -->

현재 상태: 이미지 기반 언어 생성은 아직 실용 수준을 검증하지 못했습니다. 최근 디코더 손실 비교 실험에서 학습에 사용한 32개 질문의 자율 생성 답변은 두 모델 모두 픽셀 완전 일치가 0/32였습니다. 디코더 민감도를 반영한 손실은 전경 F1을 대조군의 85.984%에서 70.009%로 낮췄습니다. 올바른 답변 이력을 제공해도 각각 92.430%와 75.116%이며, 정답 이미지 코드를 직접 제공한 재구성은 양쪽 모두 99.9953%입니다. 수정 기준 네 항목을 모두 통과하지 못했으므로 이 학습 방식의 규모를 확대하지 않습니다. 이는 이미지 일치도이며 의미 정확도가 아닙니다. 사용 가능성을 검증한 ILM 체크포인트는 아직 없습니다. [결과 및 모든 원본 이미지](../references/decoder_metric_v82_result.md). 논문에는 보충 노트 39와 편집 가능한 그림 S16이 반영되었습니다. 외부 이미지 생성기를 사용한 별도 기준 모델도 로컬 학습 없이 OCR 완전 일치가 복사 요청 0/12, 질문 0/8이었습니다. 참조 이미지 OCR 보정은 18/20이며 원본 출력 40개를 모두 보존했습니다. [FLUX.2-klein-4B](../references/image_edit_baseline_result.md).



[![LazyingArt banner — apps and four buddies](https://lazying.art/media/brand/lazyingart-intro.gif)](https://lazying.art)

# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Paradigm](https://img.shields.io/badge/paradigm-predictive%20visual%20field-16835B)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Domain](https://img.shields.io/badge/domain-historic%20etymology%20%7C%20glyph%20models-2F80ED?logo=github)

ILM은 언어를 **보이는 글쓰기 이미지**로 학습하고 생성합니다. 이미지에서 연속
시각 상태로, 다시 이미지로 이어지는 경로를 연구하며 숨은 기호 채널을 두지 않습니다.


최근 인터페이스 검증에서 같은 페이지를 1024에서 432로 축소하자 동결된 사전 학습 모델의 영어 정답 완전 일치가 0/32에서 11/32로 높아졌습니다. 학습한 두 어댑터는 432에서도 각각 5/32에 그쳤습니다. 답변을 이미지로 생성하는 독립 ILM은 아직 입증되지 않았습니다.

[2026-11-07: English result and full evidence](../references/native_reader_geometry_result.md)

2026-11-08 결과: 128회 업데이트한 학습의 평가를 완료하고 원본 답변 이미지 384장을 모두 검토했습니다. 초기 모델과 최종 모델 모두 학습한 질문과 보류한 질문의 각 그룹에서 OCR 완전 일치가 0/32이며, 원본에서 허용한 대체 정답을 포함해도 같습니다. 보류 그룹의 정확한 베껴 쓰기는 19/32에서 16/32로 떨어졌습니다. 사전에 정한 여섯 조건을 모두 충족하지 못했으며, 언어 학습은 아직 입증되지 않았습니다.

[결과와 근거의 범위](../references/native_paired_qa_learning_result.md)

## 고전 중국어 문장 이미지를 읽고 현대 중국어로 쓰기

매개변수 약 203만 개의 모델은 학습에서 제외한 장을 평가했을 때 이미지로 만들 수 있는 원문 2,045개를 모두 정확히 베꼈다. 현대 중국어 번역은 2,048개 중 30개가 참조 이미지와 일치했다. 한 예에서는 글자꼴 아홉 개를 복사하고 두 개를 새로 그려 薨을 去世로 바꿨다. 전체 글자꼴 편집 거리에서는 단순한 국소 편집 기법이 더 우수했다. [모델 실행](../models/visual_edit_zh/README.md) · [출력과 참조 이미지](../publication/ilm-nature/figures/Figure_48_visual_edit.png)

## 자료에서 옛 글자꼴 읽기

네 줄로 된 자료를 제시했을 때 모델은 훈련에서 제외한 글자 계열에 관한 중국어·영어 질문 9,600개 중 9,565개에 요청한 글자꼴을 정확한 순서로 답했습니다. [모델 실행](../models/pictured_idiom_zh/README.md) · [Fig. 42](../publication/ilm-nature/figures/Figure_42_historical_plates.png)

## 앞뒤 문장을 이미지로 읽고 빠진 중국어 성어 쓰기

매개변수 212만 개의 모델이 빈칸 앞뒤의 문장 이미지를 읽고 빠진 성어를 한 글자씩 그립니다. 검증 사례 22,907개 중 134개에서 원문과 정확히 일치했습니다. 그중 100개는 정답의 네 글자 모두가 입력에 없었습니다. 주변 글자의 출현 빈도를 이용한 기준 모델은 1,012개를 맞혔습니다.

[모델 실행](../models/pictured_idiom_zh/README.md) · [그림 40](../publication/ilm-nature/figures/Figure_40_pictured_idioms.png)

읽기 모델의 예측을 추가 학습에 사용하자 22,907개 중 378개가 원문과 일치했습니다. 원래 정답만으로 같은 횟수만큼 추가 학습한 경우에는 292개였습니다. 별도로 확보한 미사용 지문 9,135개에서는 각각 163개와 125개를 맞혔습니다. [그림 41](../publication/ilm-nature/figures/Figure_41_writer_transfer.png)

시각 판독기와 생성기를 함께 학습한 모델은 새로운 중국어 빈칸 17,555개 중 433개에 정확한 답을 그렸습니다. 이전 모델은 276개, 인접 글자 빈도 기준선은 789개를 맞혔습니다. [Fig. 43](../publication/ilm-nature/figures/Figure_43_expression_preference.png)

## 중국어 글자 이미지에서 답 고르기

소형 독해 모델을 CPU 스레드 2개로 64초 동안 학습했습니다. 학습에 쓰지 않은 중국어 지문의 질문 1,788개 중 269개에서 정답과 픽셀 단위로 일치하는 답을 골랐습니다. 질문을 빈 이미지로 바꾸자 정답은 18개로 줄었습니다. 별도의 자유 생성 모델은 言의 변천사를 읽을 수 있는 형태로 설명하지 못했습니다. 범용 글쓰기 능력과 1,000배의 학습 효율은 아직 입증되지 않았습니다.

![중국어 글자 이미지에서 답 고르기](../publication/ilm-nature/figures/Figure_34_visual_reading_writing.png)

[독해 모델 실행](../references/visual_extractive_reading_result.md) · [논문](../publication/ilm-nature/manuscript.pdf)

중국어 질문 3,170개 중 565개에서 읽기 모델이 올바른 본문 구간을 골랐다. 쓰기 모델은 이 가운데 124개 답 전체를 다른 서체로 정확히 그렸다. 같은 밴드에 대한 두 질문에는 각각 前卫金属와 1996年을 썼다.

![다른 서체로 쓴 답](../publication/ilm-nature/figures/Figure_38_grounded_detail.png)

[다른 서체로 쓴 답](../references/grounded_detail_writing_result.md)

## 문맥을 보고 다음 한자 그리기

개발용 문맥 668개에서 글자 전체의 대략적인 획 배치를 먼저 예측하자 공백이 아닌 기존 글자 모양과 픽셀 단위로 일치한 출력이 198개에서 649개로 늘었다. 원문에 실제로 이어지는 글자를 맞힌 횟수는 두 모델 모두 18회였다. 言의 유래를 묻자 새 모델은 숫자 0을 닮은 모양을 반복했다.

![문맥을 보고 다음 한자 그리기](../publication/ilm-nature/figures/Figure_37_shared_layout.png)

[실험 결과와 학습된 모델](../references/shared_layout_writing_result.md)

## 이전 이어쓰기 실험

2026-11-13

전체 시퀀스 실험은 **13,766개 시퀀스, 41,298회 업데이트, 6,108,741개 목표 글리프 위치**를 처리했습니다. 참조 정답으로 채점하기 전에 **출력 이미지 732장 전부**를 검토했습니다. 최종 모델의 비례 배치 입력 결과는 **엄격한 완전 정답 7/32, 같은 지문 두 질문 모두 정답 0/16쌍, 읽을 수 있는 답변 14/32, 일관되고 완결된 긴 책 이어쓰기 0/12**입니다. 사전 등록 기준과 질문·지문을 지우는 네 가지 대조군에 대한 독립 감사 모두 통과하지 못했습니다.

RTX 4090 한 장으로 학습했으며 최대 할당 VRAM은 **18.568 GiB**입니다. 물려받은 기반 가중치를 포함한 전체 실행 모델은 **7,578,390,003개 파라미터**이고, 이번 학습에서 갱신한 파라미터는 **20,212,225개**입니다. 재사용한 개발 데이터와 한 명의 모델 평가자가 사용되었으며 독립적인 사람의 판정은 없습니다. 실용적인 경량 ILM과 동일 품질에서의 효율 우위는 아직 입증되지 않았습니다.

[전체 결과](../references/visual_continuation_learning_result.md) · [개발 목표](../docs/first-imagized-language-model-goal.md) · [고정 실험 규약](../references/visual_continuation_full_protocol.md)

## 이전 V35 결과

![V35 실측 결과: 완전한 래스터 입력 및 출력 모델이 프롬프트에 반응하는 흔적을 생성했지만 복사와 지시 결합에는 실패함](../publication/ilm-image-native/figures/causal_glyph_flow_v35_result.png)

V35는 `129,092,738`개 파라미터의 중국어 시각 언어 모델입니다. RTX 4090 한 장에서
BF16 업데이트 `22,000`회를 모두 완료했으며, `2.770`시간과 최대 `2.899 GiB`
GPU 메모리를 사용했습니다. 배포 경로는 글쓰기 픽셀과 시각 마스크만 받아 래스터를
직접 생성하고 다시 읽습니다. 실행 중 tokenizer, Unicode/문자 ID, OCR, 검색,
시각 코드북 또는 교사 모델을 사용하지 않습니다.

실측 판정은 **`not-qualified`**입니다. 올바른 프롬프트의 복사 정확도는
`0.3125%`, 지시 정확도는 `0.1116%`이며 지시 정확도는 섞인 프롬프트 대조군보다
낮았습니다. 시각 인과 게이트는 `5/12`, 의미 래스터 게이트는 `5/9`만 통과했습니다.
출력은 비어 있지 않고 프롬프트 변화에 반응하지만 글자 모양이 손상되어 요청한 답과
결합되지 않았습니다. 따라서 봉인된 분할은 열지 않았습니다.

V35는 전이 학습을 사용합니다. V34 연속 글쓰기 코덱은 이 프로젝트의 작업이지만,
인과 코어는 외부 PIXAR 체크포인트로 초기화했습니다. PIXAR는 외부 기반으로 명확히
표기하며 이 프로젝트의 기여로 주장하지 않습니다. 여기서 “독립”은 자체 완결된 배포
산출물을 뜻하며 모든 구성 요소를 처음부터 학습했다는 뜻이 아닙니다. 자세한 내용은
[영문 실측 보고서](../references/causal_glyph_flow_v35_result.md), [구현 안내](../docs/causal-glyph-flow-v35.md),
[논문](../publication/ilm-image-native/ilm-image-native.pdf)을 참조하십시오.

## 이전 제한 프롬프트 실험: V23 시각 관계 회로 통과

![V23 실측 결과: 여섯 이미지 프롬프트가 시각적 일치, 연산 게이트, 원본 글리프 라우팅, 고정 이미지 정규화기를 거쳐 한 장의 답변 이미지로 출력됨](../publication/ilm-image-native/figures/visual_relation_circuit_v23_result.png)

V23은 이 저장소에서 전체 증거 절차를 통과한 첫 이미지-프롬프트 입력 및
이미지-답변 출력 실험입니다. 학생 모델은 여섯 장의 `32x32` 글쓰기 이미지만 받고
한 장의 `32x32` 답변 이미지를 출력합니다. 배포 경로에는 문자열, token, Unicode ID,
OCR, 글리프 조회, 답변 인덱스, 외부 언어 모델이 없습니다.

단 한 번 허가된 동결 평가에서 보지 못한 한자 98자, 1,024개 에피소드, 4,096개 프롬프트
변형에 대해 이진 선택 `0.99829`, 질의 전환 `0.99609`, 연산 전환 `0.99707`, 출력
글리프 top-1 `0.99463`, pixel F1 `0.78478`을 얻었습니다. 질의 맹검 및 연산 맹검
대조군은 각자 볼 수 없는 요인에 대해 정확히 전환 0을 유지합니다.

이 결과는 고정된 여섯 역할, 두 쌍의 결합, same/other 관계에 한정된 시각 프롬프트
추종을 입증하며 자유 형식 언어 이해를 뜻하지 않습니다. V24는 고정 프레임 역할을 없애고
가변 길이 2D 글쓰기 이미지 스트림을 읽은 뒤 첫 출력을 다시 읽어 두 번째 프레임을
생성합니다. 전체 증거는 [영문 V23 결과 기록](../docs/visual-relation-circuit-v23-result.md)에 있습니다.

## 이전 연구 기준: Predictive Visual Field의 출발점

![Predictive Visual Field 구조](../publication/ilm-image-native/figures/predictive_visual_field_paradigm.png)

RFLM V7은 하나의 픽셀 흐름이 언어적 정체성을 추론하면서 획까지 동시에 그리도록 한
구조의 한계를 보였습니다. 전체 문맥 top-1은 `1.20%`에서 `2.31%`로 올라
last-only (`2.02%`)와 unigram (`1.86%`)을 넘었지만 bigram (`13.58%`)에는
크게 못 미쳤습니다. 정규화 로그확률 이득도 `-0.9066`에서 `-0.2155`로
개선됐지만 여전히 음수이고 자율 출력은 읽을 수 없습니다. 따라서 **V7은 언어
모델로서 불합격**입니다.

다음 PVF는 다음 망막 상태를 예측하는 연속 흐름과 그 상태를 잉크로 렌더링한 뒤
다시 읽는 시각 액추에이터를 분리합니다. 최근접 문자 검색이나 출력 테이블은 없습니다.
경계는 `글쓰기 픽셀 -> 연속 시각 동역학 -> 잉크 픽셀`이며 token, Unicode ID,
OCR, 시각 코드북, 외부 언어 모델을 받지 않습니다. PVF는 검증 가능한 V8 가설이지
입증된 능력이 아닙니다.

> 저장소는 실용적인 어원 파이프라인과 장기 ILM 실험을 나란히 유지합니다.

## 📌 개요

이 저장소에는 세 개의 연결된 트랙이 있습니다.

1. Retinal Flow 이미지 네이티브 언어 모델링과 엄격한 외부 평가.
2. 출처를 보존하는 역사적 중국어 글리프 어원 수집.
3. 재현성을 위해 보존한 이전 글리프, 코드북, 확산, folio, InkStream 기준선.

이 README는 세 트랙을 문서화하며 어원 워크플로를 재현 가능한 핵심 경로로 유지합니다.

## 🔗 주요 링크

| 영역 | 경로 |
|---|---|
| 개념 정리 문서 | `docs/imagized-language-model.md` |
| 현재 엔지니어링 목표 | `docs/first-imagized-language-model-goal.md` |
| 연구 자료와 실험 근거 | `references/image-native-language-model-research.md` |
| 코드 계획 및 지표 | `docs/ilm-visual-diffusion-code-plan.md` |
| 임베딩 "색상" 계획 | `docs/embedding-color-plan.md` |
| 개발 노트/계획 | `docs/development-plan.md` |
| 어원 모듈 README | `ilm/etymology/README.md` |

## ✨ 기능

- 🏺 `hanziyuan` 및 `chineseetymology` 계열 소스에서 어원 수집.
- 🌐 재시도, 속도 제한, 캐시가 포함된 강건한 AJAX + HTML 수집 경로.
- 🧩 `<img>` 및 CSS `background-image` data URI를 포함한 단계 라벨 글리프 추출.
- 🗃️ 문자/글리프 메타데이터 및 파일 시스템 자산 배치를 위한 SQLite 기반 저장소.
- 🖥️ 임시 수집 + 갤러리 미리보기를 위한 Tornado 웹 UI.
- 🔤 다국어 토큰 이미지용 글리프 렌더링 유틸리티.
- 🧠 제품 코드 스타일 임베딩/코드북 모듈.
- 🧱 문장 프레임 패킹 및 확산/인페인팅 학습·평가 스크립트.
- 📊 임베딩 및 파이프라인 점검용 리포팅/시각화 스크립트.
- 📄 `publication/` 아래 LaTeX/PDF 출판물 아티팩트.

## 🧱 프로젝트 구조

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

## 🧰 필수 조건

| 요구 사항 | 참고 |
|---|---|
| Python `3.10+` | 핵심 런타임 |
| `pip` | 패키지 설치 |
| 선택적 GPU | PyTorch CUDA 학습 스크립트에 유용 |
| 선택적 LaTeX 툴체인 | 출판물 빌드에 필요 |

가정 메모: 현재 루트에 단일한 의존성 잠금/명세 파일(`pyproject.toml`, `requirements.txt` 등)이 없으므로, 의존성은 import와 스크립트 사용 방식에서 추론됩니다.

## ⚙️ 설치

### 최소(어원 툴킷)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### 확장(모델링/학습 워크플로우)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

특정 스크립트가 추가 패키지를 필요로 한다면 해당 스크립트에서 발생한 import 에러를 보고 설치하세요.

## 🚀 사용법

### 빠른 시작: 역사적 글리프 수집(CLI)

1. Hanziyuan(권장): 문자 전용 AJAX 흐름

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology(직접 URL)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. 배치 파일 수집(행 형식: `char\turl`, `url`, `char url`)

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

`http://127.0.0.1:8888`을 열고 사이트를 선택한 뒤, 문자(예: `中`)를 입력합니다.

### 예의 있는 크롤링과 사이트 존중

- 페처는 호스트별 속도 제한, 지수 백오프 재시도, 캐시를 사용합니다.
- 지연을 `>= 0.5s`로 유지하고 요청 폭주를 피하며, 사이트 약관/robots/라이선스를 준수하세요.
- paywall나 인터랙티브 방어를 우회하지 마세요.
- `403`/`429`가 나타나면 속도를 늦추고 나중에 다시 시도하세요.

### 추가 ILM 워크플로우

아래 스크립트는 저장소에 존재하며 실무적으로 활용되고 있지만, 연구 워크플로우이므로 준비된 로컬 데이터셋/체크포인트가 필요할 수 있습니다.

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

## 🧩 설정

주요 YAML 구성:

- `configs/color.yaml`
  - 데이터 경로: `data/processed/images_common_freq/index.tsv`
  - 모델/코드 파라미터: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - optimizer/log 설정

- `configs/diffusion.yaml`
  - 입력 JSONL: `data/processed/test_100.jsonl`
  - 프레임/그리드 + 모델 크기 설정
  - 학습 마스크 비율 범위 및 체크포인트 설정

지원되는 경우 CLI 플래그(`--epochs`, `--batch-size`, `--lr` 등)로 설정을 덮어쓸 수 있습니다.

## 🧪 예시

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

- Hanziyuan에서 공통 문자 대량 수집:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 📝 개발 노트

- 이 저장소는 탄탄한 CLI와 탐색적 산출물(노트북 및 프로토타입 스크립트 포함)을 함께 다루는 연구 저장소입니다.
- 대용량 산출물은 `data/` 및 `artifacts/`에 두는 것을 전제로 하며(두 위치 모두 `.gitignore`에 포함).
- 출판 소스와 PDF는 `publication/` 아래 있고, 보조 빌드 스크립트는 `scripts/latex_build.sh`입니다.
- 협업/프로세스 규범은 `AGENTS.md`에 문서화되어 있습니다.

## 🛠️ 문제 해결

- `ModuleNotFoundError: ilm...`
  - 저장소 루트에서 스크립트를 실행하세요.
  - 로컬 패키지 해석이 필요한 스크립트는 `PYTHONPATH=.`를 사용하세요.

- 데이터/인덱스/체크포인트에 대한 `FileNotFoundError`
  - 먼저 필수 데이터/빌드 스크립트를 실행하세요.
  - `data/processed/images_common_freq/index.tsv`, `data/processed/test_100.jsonl`과 같은 기본 경로가 존재하는지 확인하세요.

- CUDA/디바이스 이슈
  - 스크립트 플래그나 설정(`device: cpu` 또는 `--device cpu`)으로 CPU로 전환하세요.

- 패키지 누락 에러
  - 특정 스크립트에서 요구하는 의존성(`torch`, `pyyaml`, `Pillow` 등)을 해당 스크립트의 import 경로에 맞춰 설치하세요.

- 스크래핑 중 HTTP `403` / `429`
  - `--delay`를 늘리고 나중에 재시도하며 예의 바른 요청 속도를 유지하세요.

## 🗺️ 로드맵

- 어원 중심 빠른 시작을 넘어 텍스트-이미지 ILM 학습/평가 런북을 계속 고도화합니다.
- 환경 재현성 개선(단일 권위 의존성 명세).
- 연구 스크립트와 파이프라인 연결부에 대한 테스트/CI 범위 확대.
- 계층형 코드북, 확산 목적 함수, 제어 채널을 반복 개선.
- `docs/`, 스크립트 도움말, 출판 아티팩트를 가로질러 문서 일관성 통합.

깊은 개념 및 단계적 계획은 아래를 참조하세요.

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 🤝 Contributing

- `AGENTS.md`의 규칙을 따르세요 (`원자적 커밋`, 변경 후 `push`, 코드 내 자격 증명 금지).
- 관련 있는 변경은 주제별로 묶어 컨벤셔널 메시지로 커밋하세요.
- 재현 가능한 스크립트 실행은 명시적인 플래그/입력 경로를 우선하세요.
- 스크래핑 관련 변경에서는 스로틀링/캐시 동작과 사이트 존중 규칙을 유지하세요.

## ❤️ Support

<!-- support:start -->
[![GitHub Sponsors](https://img.shields.io/badge/GitHub-Sponsors-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)
[![LazyingArt Support](https://img.shields.io/badge/LazyingArt-Support-0EA5E9?style=for-the-badge)](https://chat.lazying.art/donate)
[![PayPal Support](https://img.shields.io/badge/PayPal-Support-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen)
[![Stripe Support](https://img.shields.io/badge/Stripe-Support-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400)
<!-- support:end -->

## 📄 License

현재 이 저장소에는 최상위 라이선스 파일이 없습니다.

가정 메모: 관리자가 `LICENSE` 파일을 추가할 때까지, 이 프로젝트는 라이선스가 미정된 연구 코드로 취급하세요.
