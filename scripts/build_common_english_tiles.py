#!/usr/bin/env python3
"""Batch-generate tiled glyph images for common English words."""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from ilm.english_tiles import (
    DEFAULT_SPEC,
    SHORT_WORD_SPEC,
    GridSpec,
    render_word_tile_image,
)

DEFAULT_COMMON_WORDS: List[str] = [
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "i",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
    "back",
    "after",
    "use",
    "two",
    "how",
    "our",
    "work",
    "first",
    "well",
    "way",
    "even",
    "new",
    "want",
    "because",
    "any",
    "these",
    "give",
    "day",
    "most",
    "us",
    "thing",
    "those",
    "tell",
    "man",
    "should",
    "child",
    "world",
    "school",
    "still",
    "try",
    "last",
    "ask",
    "need",
    "too",
    "feel",
    "three",
    "state",
    "never",
    "become",
    "between",
    "high",
    "really",
    "something",
    "another",
    "much",
    "family",
    "own",
    "leave",
    "put",
    "old",
    "while",
    "mean",
    "keep",
    "student",
    "why",
    "let",
    "great",
    "same",
    "big",
    "group",
    "begin",
    "seem",
    "country",
    "help",
    "talk",
    "where",
    "turn",
    "problem",
    "every",
    "start",
    "hand",
    "might",
    "american",
    "show",
    "part",
    "against",
    "such",
    "again",
    "few",
    "case",
    "week",
    "company",
    "system",
    "each",
    "right",
    "program",
    "hear",
    "question",
    "during",
    "play",
    "government",
    "run",
    "small",
    "number",
    "off",
    "always",
    "move",
    "night",
    "live",
    "point",
    "believe",
    "hold",
    "today",
    "bring",
    "happen",
    "next",
    "without",
    "before",
    "large",
    "million",
    "must",
    "home",
    "under",
    "water",
    "room",
    "write",
    "mother",
    "area",
    "national",
    "money",
    "story",
    "young",
    "fact",
    "month",
    "different",
    "lot",
    "study",
    "book",
    "eye",
    "job",
    "word",
    "though",
    "business",
    "issue",
    "side",
    "kind",
    "four",
    "head",
    "far",
    "black",
    "long",
    "both",
    "little",
    "house",
    "yes",
    "since",
    "provide",
    "service",
    "around",
    "friend",
    "important",
    "father",
    "sit",
    "away",
    "until",
    "power",
    "hour",
    "game",
    "often",
    "yet",
    "line",
    "political",
    "end",
    "among",
    "ever",
    "stand",
    "bad",
    "lose",
    "however",
    "member",
    "pay",
]


def load_words(path: str | None) -> List[str]:
    if path is None:
        return DEFAULT_COMMON_WORDS
    words: List[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token and not token.startswith("#"):
            words.append(token)
    return words


def safe_filename(word: str, suffix: str, *, used: set[str]) -> Path:
    slug_chars = []
    for ch in word.lower():
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            slug_chars.append(ch)
        elif ch in {"-", "_"}:
            slug_chars.append(ch)
        elif ch.isspace():
            slug_chars.append("_")
    slug = "".join(slug_chars).strip("_") or "word"
    candidate = slug
    counter = 1
    while candidate in used:
        counter += 1
        candidate = f"{slug}_{counter}"
    used.add(candidate)
    return Path(f"{candidate}{suffix}")


def save_images(
    words: Iterable[str],
    out_dir: Path,
    spec: GridSpec,
    short_spec: GridSpec,
    *,
    dynamic: bool = True,
) -> None:
    used_names: set[str] = set()
    index_lines = []
    for word in words:
        image = render_word_tile_image(
            word,
            spec=spec,
            short_spec=short_spec,
            dynamic=dynamic,
        )
        png_path = safe_filename(word, ".png", used=used_names)
        full_png = out_dir / png_path
        full_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(full_png)
        index_lines.append(f"{word}\t{png_path.as_posix()}\n")
    (out_dir / "index.tsv").write_text("".join(index_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate glyph grids for common English words")
    parser.add_argument(
        "--word-list",
        help="Optional newline-delimited list of words to render",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/english_common_tiles",
        help="Directory to store rendered PNGs (default: artifacts/english_common_tiles)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_SPEC.tile_size,
        help="Tile size for the main layout (default: 16)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_SPEC.grid_size,
        help="Grid size for the main layout (default: 8)",
    )
    parser.add_argument(
        "--short-tile-size",
        type=int,
        default=SHORT_WORD_SPEC.tile_size,
        help="Tile size for short words (default: 32)",
    )
    parser.add_argument(
        "--short-grid-size",
        type=int,
        default=SHORT_WORD_SPEC.grid_size,
        help="Grid size for short words (default: 4)",
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Disable dynamic short-word scaling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    words = load_words(args.word_list)
    spec = GridSpec(grid_size=args.grid_size, tile_size=args.tile_size)
    short_spec = GridSpec(grid_size=args.short_grid_size, tile_size=args.short_tile_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_images(
        words,
        out_dir,
        spec,
        short_spec,
        dynamic=not args.no_dynamic,
    )
    print(f"Rendered {len(words)} words into {out_dir}")


if __name__ == "__main__":
    main()
