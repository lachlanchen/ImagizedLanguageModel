#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def write_lines(path: Path, iterable, limit: int, field: str = "text", lang: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in iterable:
            txt = ex.get(field, "").strip()
            if not txt:
                continue
            f.write((json.dumps({"text": txt, "lang": lang}) + "\n"))
            n += 1
            if 0 < limit <= n:
                break
    return n


def main():
    ap = argparse.ArgumentParser(description="Download small EN/ZH corpora to JSONL")
    ap.add_argument("--out", default="data/raw", help="output folder")
    ap.add_argument("--en_source", default="mc4", choices=["mc4", "wikitext"], help="EN dataset")
    ap.add_argument("--zh_source", default="mc4", choices=["mc4"], help="ZH dataset")
    ap.add_argument("--en_limit", type=int, default=20000, help="EN documents limit")
    ap.add_argument("--zh_limit", type=int, default=20000, help="ZH documents limit")
    ap.add_argument("--bitext", action="store_true", help="also download small EN-ZH parallel set (opus_books)")
    ap.add_argument("--bitext_limit", type=int, default=20000, help="bitext pairs limit")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import to allow script to show helpful errors
    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit("Please install 'datasets' (pip install datasets). Error: %s" % e)

    # English
    if args.en_source == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        en_iter = ( {"text": ex["text"]} for ex in ds["train"] )
        en_written = write_lines(out_dir / "en.jsonl", en_iter, args.en_limit, field="text", lang="en")
    else:
        ds = load_dataset("mc4", "en", streaming=True)
        en_iter = ( {"text": ex["text"]} for ex in ds["train"] )
        en_written = write_lines(out_dir / "en.jsonl", en_iter, args.en_limit, field="text", lang="en")

    # Chinese
    if args.zh_source == "mc4":
        dszh = load_dataset("mc4", "zh", streaming=True)
        zh_iter = ( {"text": ex["text"]} for ex in dszh["train"] )
        zh_written = write_lines(out_dir / "zh.jsonl", zh_iter, args.zh_limit, field="text", lang="zh")
    else:
        zh_written = 0

    # Optional bitext (very small sample for alignment)
    bitext_written = 0
    if args.bitext:
        try:
            opus = load_dataset("opus_books", "en-zh", split="train")
            outp = out_dir / "bitext.tsv"
            outp.parent.mkdir(parents=True, exist_ok=True)
            with outp.open("w", encoding="utf-8") as f:
                for i, ex in enumerate(opus):
                    en = (ex.get("translation", {}) or {}).get("en", "").strip()
                    zh = (ex.get("translation", {}) or {}).get("zh", "").strip()
                    if en and zh:
                        f.write(en.replace("\t", " ") + "\t" + zh.replace("\t", " ") + "\n")
                        bitext_written += 1
                        if 0 < args.bitext_limit <= bitext_written:
                            break
        except Exception:
            pass

    meta = {
        "en_source": args.en_source,
        "zh_source": args.zh_source,
        "en_written": en_written,
        "zh_written": zh_written,
        "bitext_written": bitext_written,
    }
    with (out_dir / "_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

