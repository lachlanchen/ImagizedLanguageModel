from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = (
    "char",
    "query_zh",
    "query_en",
    "answer_zh",
    "answer_en",
    "classical_style",
)

STAGE_NAMES_ZH = {
    "oracle": "甲骨文",
    "bronze": "金文",
    "seal": "小篆",
    "liushutong": "六书通字形",
}

STAGE_NAMES_EN = {
    "oracle": "oracle-bone",
    "bronze": "bronze-inscription",
    "seal": "small-seal",
    "liushutong": "Liushutong",
}


@dataclass(frozen=True)
class HistoricalTeacherRecord:
    char: str
    query_zh: str
    query_en: str
    answer_zh: str
    answer_en: str
    classical_style: str
    stage_counts: dict[str, int]
    teacher_model: str
    teacher_endpoint: str
    latency_seconds: float
    source_scope: str = "semantic wording only; visual claims require local glyph evidence"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TeacherResponseError(RuntimeError):
    pass


def _extract_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise TeacherResponseError("teacher did not return a JSON object") from error
        try:
            parsed = json.loads(content[start : end + 1])
        except json.JSONDecodeError as nested_error:
            raise TeacherResponseError("teacher returned malformed JSON") from nested_error
    if not isinstance(parsed, dict):
        raise TeacherResponseError("teacher JSON must be an object")
    return parsed


def _validated_plan(plan: dict[str, Any], expected_char: str) -> dict[str, str]:
    missing = [field for field in REQUIRED_FIELDS if not isinstance(plan.get(field), str) or not plan[field].strip()]
    if missing:
        raise TeacherResponseError(f"teacher fields missing or empty: {missing}")
    cleaned = {field: plan[field].strip() for field in REQUIRED_FIELDS}
    if cleaned["char"] != expected_char:
        raise TeacherResponseError(
            f"teacher returned char={cleaned['char']!r}, expected {expected_char!r}"
        )
    combined = " ".join(cleaned.values())
    forbidden = ("《", "》", "说文解字", "説文解字", "据文献记载", "柳体", "隶书")
    if any(marker in combined for marker in forbidden):
        raise TeacherResponseError("teacher introduced an unsupported citation")
    limits = {
        "query_zh": 80,
        "query_en": 180,
        "answer_zh": 140,
        "answer_en": 260,
        "classical_style": 50,
    }
    for field, limit in limits.items():
        if len(cleaned[field]) > limit:
            raise TeacherResponseError(f"teacher field {field} exceeds {limit} characters")
    return cleaned


class LocalLLMTeacher:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8008/v1",
        api_key: str = "local-dev-key",
        model: str = "qwen3:8b-q4_K_M",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def historical_plan(
        self,
        char: str,
        stage_counts: dict[str, int],
        *,
        retries: int = 2,
    ) -> HistoricalTeacherRecord:
        stage_summary = "、".join(
            f"{STAGE_NAMES_ZH.get(stage, stage)}:{count}幅"
            for stage, count in stage_counts.items()
            if count > 0
        )
        system = (
            "你是文字学训练数据编辑，但你看不到具体字形图。只输出一个合法JSON对象。"
            "不得编造字形细节、年代、出处、引文或《说文》内容。答案只能说明将依据本地实物图像比较各阶段，"
            "不得把六书通写成柳体或隶书。classical_style只能写无引号的仿古短句，不得冒充古籍原文。/no_think"
        )
        user = (
            f"为书写字符“{char}”制作双语图像问答页语义计划。可用的本地图像数量为：{stage_summary}。"
            "这些数量只证明有图，不证明具体构形。返回且仅返回字段："
            "char,query_zh,query_en,answer_zh,answer_en,classical_style。"
            "中英文问题都询问该汉字或Kanji的有证据字形源流；答案提醒读者逐图辨析，文字简洁。/no_think"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 900,
            "stream": False,
            "think": False,
            "response_format": {"type": "json_object"},
        }
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            started = time.perf_counter()
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                plan = _validated_plan(_extract_json(content), char)
                stages_zh = "、".join(
                    STAGE_NAMES_ZH.get(stage, stage)
                    for stage, count in stage_counts.items()
                    if count > 0
                )
                stages_en = ", ".join(
                    STAGE_NAMES_EN.get(stage, stage)
                    for stage, count in stage_counts.items()
                    if count > 0
                )
                safe_answer_zh = (
                    f"本页仅按本地收录的{stages_zh}图像并列展示“{char}”的书写形体。"
                    "具体构形、先后关系与字义解释须逐图核对出处；生成图形不得冒充出土字例。"
                )
                safe_answer_en = (
                    f"This page juxtaposes locally catalogued {stages_en} images for {char}. "
                    "Exact morphology, chronology, and etymology require source-by-source verification; "
                    "a generated form is not an attested specimen."
                )
                return HistoricalTeacherRecord(
                    char=plan["char"],
                    query_zh=plan["query_zh"],
                    query_en=plan["query_en"],
                    answer_zh=safe_answer_zh,
                    answer_en=safe_answer_en,
                    classical_style="观形识变，考其所自。",
                    stage_counts=dict(stage_counts),
                    teacher_model=self.model,
                    teacher_endpoint=self.base_url,
                    latency_seconds=time.perf_counter() - started,
                )
            except (KeyError, TypeError, ValueError, urllib.error.URLError, TeacherResponseError) as error:
                last_error = error
                if attempt < retries:
                    time.sleep(1.0 + attempt)
        raise TeacherResponseError(f"teacher failed for {char!r}: {last_error}")


def load_teacher_manifest(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed teacher JSONL at line {line_number}") from error
        char = item.get("char")
        if isinstance(char, str) and char:
            records[char] = item
    return records


def save_teacher_manifest(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)
