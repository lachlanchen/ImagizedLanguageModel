from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple, Dict

from ilm.db.glyph_db import GlyphDB

# Special tokens (language-scoped)
BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"


_WORD_RE = re.compile(r"[A-Za-z]+|\d+|[^\w\s]", re.UNICODE)


def tokenize_en(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def tokenize_zh(text: str) -> List[str]:
    out: List[str] = []
    for ch in text:
        if _is_cjk(ch):
            out.append(ch)
    return out


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
        or 0x2F800 <= code <= 0x2FA1F
    )


@dataclass
class QAPair:
    lang: str  # 'en' or 'zh'
    q_tokens: List[str]
    a_tokens: List[str]


def load_alpaca_json(path: str | Path) -> List[dict]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    assert isinstance(data, list)
    return data


def iter_alpaca_qa(
    path: str | Path, lang: str, *, use_input: bool = True
) -> Iterator[QAPair]:
    tok = tokenize_en if lang == "en" else tokenize_zh
    for rec in load_alpaca_json(path):
        instr = rec.get("instruction", "")
        inp = rec.get("input", "") if use_input else ""
        out = rec.get("output", "")
        q_text = (instr + "\n" + inp).strip()
        a_text = out.strip()
        if not q_text or not a_text:
            continue
        q_toks = tok(q_text)
        a_toks = tok(a_text)
        if q_toks and a_toks:
            yield QAPair(lang=lang, q_tokens=q_toks, a_tokens=a_toks)


def build_vocab_and_cache_glyphs(
    qa_pairs: List[QAPair], db: GlyphDB, glyph_size: int = 128
) -> Tuple[Dict[str, int], List[QAPair], List[str]]:
    """Build a vocab per-language and pre-render glyphs into the DB.

    Returns a token->index dict scoped by (lang, token), and the original pairs.
    """
    vocab: dict[tuple[str, str], int] = {}
    next_id = 0
    languages = set()
    for pair in qa_pairs:
        languages.add(pair.lang)
        for token in pair.q_tokens + pair.a_tokens:
            key = (pair.lang, token)
            if key not in vocab:
                vocab[key] = next_id
                next_id += 1
                # Ensure glyph on disk
                db.ensure_glyph(pair.lang, token, glyph_size)
    # Add language-scoped special tokens and ensure glyphs
    for lang in sorted(languages):
        for tok in (BOS, EOS, PAD):
            key = (lang, tok)
            if key not in vocab:
                vocab[key] = next_id
                next_id += 1
                db.ensure_glyph(lang, tok, glyph_size)
    # Remap to a flat map (lang::token -> id)
    flat_vocab = {f"{k[0]}::{k[1]}": v for k, v in vocab.items()}
    # id -> lang list (index by id)
    id_to_lang = [None] * len(flat_vocab)
    for (lang, tok), idx in vocab.items():
        id_to_lang[idx] = lang
    return flat_vocab, qa_pairs, id_to_lang


def special_token_ids(flat_vocab: Dict[str, int], lang: str) -> tuple[int | None, int | None, int | None]:
    bos = flat_vocab.get(f"{lang}::{BOS}")
    eos = flat_vocab.get(f"{lang}::{EOS}")
    pad = flat_vocab.get(f"{lang}::{PAD}")
    return bos, eos, pad

def id_to_lang_from_vocab(flat_vocab: Dict[str, int]) -> List[str]:
    id_to_lang = [None] * len(flat_vocab)
    for k, idx in flat_vocab.items():
        lang = k.split("::", 1)[0]
        id_to_lang[idx] = lang
    return id_to_lang
