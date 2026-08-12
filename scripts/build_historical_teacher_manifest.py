#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ilm.visual_lm.rendering import GlyphCorpus, STAGES
from ilm.visual_lm.teacher import (
    LocalLLMTeacher,
    load_teacher_manifest,
    save_teacher_manifest,
)


DEFAULT_CHARS = "言,中,水,日,月,人,山,火,木,口,学,車,车,王,雨,田,金"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Qwen teacher wording for visual historical-form tasks."
    )
    parser.add_argument("--glyph-root", default=None)
    parser.add_argument("--characters", default=DEFAULT_CHARS)
    parser.add_argument("--out", default="data/teacher/historical_qwen8b.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8008/v1")
    parser.add_argument("--api-key", default="local-dev-key")
    parser.add_argument("--model", default="qwen3:8b-q4_K_M")
    parser.add_argument("--delay", type=float, default=0.75)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.out)
    existing = load_teacher_manifest(output)
    corpus = GlyphCorpus(args.glyph_root)
    teacher = LocalLLMTeacher(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )
    characters = [item.strip() for item in args.characters.split(",") if item.strip()]
    all_records = dict(existing)
    for char in characters:
        if char in existing:
            print(json.dumps({"char": char, "status": "cached"}, ensure_ascii=False))
            continue
        if not (corpus.root / char).exists():
            print(json.dumps({"char": char, "status": "missing-glyph-directory"}, ensure_ascii=False))
            continue
        stage_counts = {}
        for stage in STAGES:
            directory = corpus.root / char / stage
            stage_counts[stage] = (
                sum(1 for path in directory.iterdir() if path.is_file()) if directory.exists() else 0
            )
        try:
            record = teacher.historical_plan(char, stage_counts, retries=args.retries)
        except Exception as error:
            print(
                json.dumps(
                    {"char": char, "status": "skipped-error", "error": str(error)},
                    ensure_ascii=False,
                )
            )
            continue
        all_records[char] = record.to_dict()
        save_teacher_manifest(output, all_records.values())
        print(
            json.dumps(
                {
                    "char": char,
                    "status": "saved",
                    "latency_seconds": record.latency_seconds,
                    "model": record.teacher_model,
                },
                ensure_ascii=False,
            )
        )
        if args.delay > 0:
            time.sleep(args.delay)
    save_teacher_manifest(output, all_records.values())
    print(json.dumps({"saved": str(output), "records": len(all_records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
