from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from ilm.utils.tokenize import tokenize_text


@dataclass
class SentenceItem:
    lang: str
    text: str
    tokens: List[str]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                continue
    return items


def pack_tokens(tokens: List[str], H: int, W: int) -> Tuple[List[str], List[int]]:
    """Truncate/pad tokens to fill an HxW grid in row-major order.
    Returns (tokens_fixed, mask_flat) with mask 1 for valid tokens, 0 for pad.
    """
    L = H * W
    tok = tokens[:L]
    mask = [1] * len(tok)
    if len(tok) < L:
        pad_n = L - len(tok)
        tok = tok + ["[PAD]"] * pad_n
        mask = mask + [0] * pad_n
    return tok, mask


class SentenceFrameDataset:
    """
    Reads a JSONL with fields {"text": str, "lang": "en"|"zh"} and yields token lists
    packed to an HxW grid. Glyph rendering and encoding is left to the training loop.
    """

    def __init__(self, jsonl_path: str, H: int = 8, W: int = 8, max_len: Optional[int] = None,
                 lang_fallback: str = "en"):
        raw = load_jsonl(jsonl_path)
        self.examples: List[SentenceItem] = []
        for obj in raw:
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            lang = (obj.get("lang") or lang_fallback).strip()
            toks = tokenize_text(text)
            if max_len is not None:
                toks = toks[:max_len]
            self.examples.append(SentenceItem(lang=lang, text=text, tokens=toks))
        self.H = H
        self.W = W

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        toks, mask = pack_tokens(ex.tokens, self.H, self.W)
        return {
            "lang": ex.lang,
            "text": ex.text,
            "tokens": toks,
            "mask": mask,
            "H": self.H,
            "W": self.W,
        }

