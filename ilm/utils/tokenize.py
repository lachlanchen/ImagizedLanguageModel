from __future__ import annotations

import re
from typing import List


_re_word = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[.,!?;:\-()\[\]{}'\"/\\]+")


def is_cjk(ch: str) -> bool:
    oc = ord(ch)
    return (
        (0x3400 <= oc <= 0x4DBF)
        or (0x4E00 <= oc <= 0x9FFF)
        or (0xF900 <= oc <= 0xFAFF)
        or (0x20000 <= oc <= 0x2A6DF)
        or (0x2A700 <= oc <= 0x2B73F)
        or (0x2B740 <= oc <= 0x2B81F)
        or (0x2B820 <= oc <= 0x2CEAF)
    )


def looks_chinese(text: str) -> bool:
    return any(is_cjk(ch) for ch in text)


def tokenize_text(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if looks_chinese(text):
        # simple char-level for CJK; keep ASCII punctuation as separate tokens
        toks: List[str] = []
        for ch in text:
            if ch.isspace():
                continue
            toks.append(ch)
        return toks
    # English-like: regex word/punct tokens
    toks = _re_word.findall(text)
    return toks

