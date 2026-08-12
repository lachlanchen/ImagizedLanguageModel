#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


WIKISOURCE_RIGHTS = "public-domain historical text; Wikisource contribution layer CC BY-SA"
WIKISOURCE_BOILERPLATE = (
    "此作品在全世界都属于公有领域",
    "本作品在全世界都属于公有领域",
    "作者逝世已经超过100年",
    "Wikisource",
)


class VisibleTextParser(HTMLParser):
    ignored_tags = {"script", "style", "nav", "table", "sup", "noscript", "math"}
    break_tags = {"p", "div", "h1", "h2", "h3", "h4", "li", "br", "blockquote", "section"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.ignored_tags:
            self.depth += 1
        elif not self.depth and tag in self.break_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.ignored_tags and self.depth:
            self.depth -= 1
        elif not self.depth and tag in self.break_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.depth:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a provenance-bearing image-language pretraining manifest."
    )
    parser.add_argument(
        "--wikisource-root",
        default="../Books/resources/curated-books/chinese-classics/public-domain-canon",
    )
    parser.add_argument("--alpaca-json", action="append", default=[])
    parser.add_argument(
        "--alpaca-rights",
        default="CC BY-NC 4.0; research use only",
        help="Recorded verbatim for optional Alpaca-derived records.",
    )
    parser.add_argument("--out", default="data/visual_grammar/chinese_mvp.jsonl")
    parser.add_argument("--chunk-chars", type=int, default=640)
    parser.add_argument("--minimum-chars", type=int, default=48)
    parser.add_argument("--maximum-records", type=int, default=None)
    return parser.parse_args()


def normalize_text(value: str) -> str:
    value = html.unescape(value).replace("\r", "\n").replace("\u3000", " ")
    value = re.sub(r"\[[0-9]+\]", "", value)
    value = re.sub(r"[\t\f\v ]+", " ", value)
    value = re.sub(r" *\n+ *", "\n", value)
    return value.strip()


def cjk_fraction(value: str) -> float:
    visible = [character for character in value if not character.isspace()]
    if not visible:
        return 0.0
    cjk = sum("\u3400" <= character <= "\u9fff" or "\U00020000" <= character <= "\U0003134f" for character in visible)
    return cjk / len(visible)


def split_chunks(text: str, maximum: int, minimum: int) -> Iterable[str]:
    text = normalize_text(text)
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    buffer = ""
    for paragraph in paragraphs:
        if any(marker in paragraph for marker in WIKISOURCE_BOILERPLATE):
            continue
        if cjk_fraction(paragraph) < 0.45:
            continue
        if buffer and len(buffer) + len(paragraph) + 1 > maximum:
            if len(buffer) >= minimum:
                yield buffer
            buffer = ""
        while len(paragraph) > maximum:
            boundary = max(paragraph.rfind(mark, 0, maximum + 1) for mark in "。！？；")
            cut = boundary + 1 if boundary >= maximum // 2 else maximum
            piece, paragraph = paragraph[:cut].strip(), paragraph[cut:].strip()
            if len(piece) >= minimum:
                yield piece
        buffer = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
    if len(buffer) >= minimum:
        yield buffer


def package_metadata(epub: Path) -> dict[str, str]:
    candidates = [
        path
        for path in epub.parent.glob("*.json")
        if path.name not in {"root.json", "checksums.json"}
    ]
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "source_url" in payload or "canonical_title" in payload:
            return {
                "title": str(payload.get("canonical_title", epub.stem)),
                "source_url": str(payload.get("source_url", "https://zh.wikisource.org/")),
                "rights": str(payload.get("rights", WIKISOURCE_RIGHTS)),
            }
    return {
        "title": epub.stem.replace("-Wikisource", ""),
        "source_url": "https://zh.wikisource.org/",
        "rights": WIKISOURCE_RIGHTS,
    }


def records_from_epub(epub: Path, *, maximum: int, minimum: int) -> Iterable[dict[str, str]]:
    metadata = package_metadata(epub)
    with zipfile.ZipFile(epub) as archive:
        entries = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith((".xhtml", ".html", ".htm"))
            and not any(marker in name.lower() for marker in ("nav", "toc", "titlepage", "cover"))
        )
        for entry in entries:
            parser = VisibleTextParser()
            parser.feed(archive.read(entry).decode("utf-8", errors="ignore"))
            for chunk_index, chunk in enumerate(split_chunks(parser.text(), maximum, minimum)):
                digest = hashlib.sha256(f"{epub}:{entry}:{chunk_index}:{chunk}".encode("utf-8")).hexdigest()[:20]
                yield {
                    "id": f"wikisource:{digest}",
                    "text": chunk,
                    "language": "zh-Hant",
                    "source": metadata["source_url"],
                    "source_title": metadata["title"],
                    "source_artifact": str(epub),
                    "rights": metadata["rights"],
                }


def records_from_alpaca(path: Path, rights: str) -> Iterable[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("instances", []))
    if not isinstance(payload, list):
        raise ValueError(f"unsupported Alpaca structure: {path}")
    for index, item in enumerate(payload):
        instruction = str(item.get("instruction", item.get("prompt", ""))).strip()
        context = str(item.get("input", item.get("context", ""))).strip()
        response = str(item.get("output", item.get("response", ""))).strip()
        text = normalize_text("\n".join(part for part in (instruction, context, response) if part))
        if len(text) < 16:
            continue
        yield {
            "id": f"alpaca:{path.stem}:{index}",
            "text": text,
            "language": "zh-Hans",
            "source": str(path),
            "source_title": path.stem,
            "source_artifact": str(path),
            "rights": rights,
        }


def main() -> None:
    args = parse_args()
    root = Path(args.wikisource_root)
    epubs = sorted(root.glob("**/*Wikisource.epub")) if root.exists() else []
    if not epubs and not args.alpaca_json:
        raise FileNotFoundError("no Wikisource EPUBs or optional Alpaca files were found")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts = {"wikisource": 0, "alpaca": 0, "duplicates": 0}
    sources: list[dict[str, object]] = []

    def emit(handle, record: dict[str, str], kind: str) -> bool:
        digest = hashlib.sha256(normalize_text(record["text"]).encode("utf-8")).hexdigest()
        if digest in seen:
            counts["duplicates"] += 1
            return False
        seen.add(digest)
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        counts[kind] += 1
        return True

    with output.open("w", encoding="utf-8") as handle:
        stop = False
        for epub in epubs:
            before = counts["wikisource"]
            for record in records_from_epub(
                epub,
                maximum=args.chunk_chars,
                minimum=args.minimum_chars,
            ):
                emit(handle, record, "wikisource")
                if args.maximum_records is not None and len(seen) >= args.maximum_records:
                    stop = True
                    break
            sources.append({"artifact": str(epub), "records": counts["wikisource"] - before})
            if stop:
                break
        if not stop:
            for value in args.alpaca_json:
                path = Path(value)
                before = counts["alpaca"]
                for record in records_from_alpaca(path, args.alpaca_rights):
                    emit(handle, record, "alpaca")
                    if args.maximum_records is not None and len(seen) >= args.maximum_records:
                        stop = True
                        break
                sources.append({"artifact": str(path), "records": counts["alpaca"] - before})
                if stop:
                    break

    provenance = {
        "schema": "ilm-visual-grammar-manifest-v1",
        "output": str(output),
        "records": len(seen),
        "counts": counts,
        "sources": sources,
        "model_boundary": "strings are used only by this offline manifest and renderer; the model receives images",
    }
    output.with_suffix(output.suffix + ".provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
