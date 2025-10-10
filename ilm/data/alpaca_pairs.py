from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset

from ilm.utils.tokenize import tokenize_text


@dataclass
class QAPair:
    lang: str
    q_tokens: List[str]
    a_tokens: List[str]


def _normalize_sample(obj: Dict) -> Tuple[str, str]:
    instr = (obj.get("instruction") or "").strip()
    inp = (obj.get("input") or "").strip()
    out = (obj.get("output") or obj.get("output_text") or "").strip()
    q = (instr + (" " + inp if inp else "")).strip()
    a = out
    return q, a


def _detect_lang(text: str, default: str = "en") -> str:
    from ilm.utils.tokenize import looks_chinese
    return "zh" if looks_chinese(text) else default


class AlpacaPairs(Dataset):
    """
    Reads Alpaca JSON file (list of dicts) and yields tokenized QA pairs.
    lang is inferred per sample (zh if contains CJK), otherwise default_lang.
    """

    def __init__(self, json_path: str, default_lang: str = "en",
                 max_len: Optional[int] = 128, min_len: int = 3):
        p = Path(json_path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        self.pairs: List[QAPair] = []
        for obj in data:
            q, a = _normalize_sample(obj)
            if not q or not a:
                continue
            lang = _detect_lang(q + a, default=default_lang)
            q_toks = tokenize_text(q)
            a_toks = tokenize_text(a)
            if max_len is not None:
                q_toks = q_toks[:max_len]
                a_toks = a_toks[:max_len]
            if len(q_toks) < min_len or len(a_toks) < min_len:
                continue
            self.pairs.append(QAPair(lang=lang, q_tokens=q_toks, a_tokens=a_toks))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> QAPair:
        return self.pairs[idx]

