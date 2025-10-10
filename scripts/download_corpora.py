#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import gzip
import io
import xml.etree.ElementTree as ET
from urllib.request import urlopen


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
    ap.add_argument("--en_source", default="wikitext", choices=["wikitext", "oscar"], help="EN dataset")
    ap.add_argument("--zh_source", default="oscar", choices=["oscar"], help="ZH dataset")
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
    en_written = 0
    try:
        if args.en_source == "wikitext":
            ds = load_dataset("wikitext", "wikitext-103-raw-v1")
            en_iter = ({"text": ex["text"]} for ex in ds["train"])
            en_written = write_lines(out_dir / "en.jsonl", en_iter, args.en_limit, field="text", lang="en")
        elif args.en_source == "oscar":
            ds = load_dataset("oscar-corpus/OSCAR-2301", "en", split="train", streaming=True)
            en_iter = ({"text": ex.get("text", "")} for ex in ds)
            en_written = write_lines(out_dir / "en.jsonl", en_iter, args.en_limit, field="text", lang="en")
    except Exception:
        en_written = 0

    # Chinese
    zh_written = 0
    try:
        if args.zh_source == "oscar":
            dszh = load_dataset("oscar-corpus/OSCAR-2301", "zh", split="train", streaming=True)
            zh_iter = ({"text": ex.get("text", "")} for ex in dszh)
            zh_written = write_lines(out_dir / "zh.jsonl", zh_iter, args.zh_limit, field="text", lang="zh")
    except Exception:
        zh_written = 0

    # Fallback: Wikipedia abstracts (ZH)
    if zh_written == 0:
        try:
            url = "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-abstract.xml.gz"
            print(f"Downloading ZH abstracts from {url} ...")
            with urlopen(url, timeout=60) as resp:
                data = resp.read()
            fh = io.BytesIO(data)
            with gzip.GzipFile(fileobj=fh) as gz:
                count = 0
                out_path = out_dir / "zh.jsonl"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    # Stream parse simplistic: read line-wise and accumulate doc
                    buf = []
                    for raw in gz:
                        line = raw.decode("utf-8", errors="ignore")
                        buf.append(line)
                        if line.strip() == "</doc>":
                            block = "".join(buf)
                            buf.clear()
                            # Extract title/abstract
                            try:
                                root = ET.fromstring(block)
                                abs_el = root.find("abstract")
                                if abs_el is not None:
                                    txt = (abs_el.text or "").strip()
                                    if txt:
                                        f.write(json.dumps({"text": txt, "lang": "zh"}, ensure_ascii=False) + "\n")
                                        count += 1
                                        if count >= args.zh_limit:
                                            break
                            except ET.ParseError:
                                continue
                zh_written = count
        except Exception:
            zh_written = 0

    # Fallback 2: CLUE TNEWS (Chinese short news titles)
    if zh_written == 0:
        try:
            clue_url = "https://raw.githubusercontent.com/CLUEbenchmark/CLUE/master/datasets/tnews/train.json"
            print(f"Downloading ZH CLUE TNEWS from {clue_url} ...")
            with urlopen(clue_url, timeout=60) as resp:
                data = resp.read().decode("utf-8", errors="ignore").splitlines()
            out_path = out_dir / "zh.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                count = 0
                for i, line in enumerate(data):
                    try:
                        obj = json.loads(line)
                        # fields can be 'sentence' or 'sentence1'
                        txt = obj.get("sentence") or obj.get("sentence1") or ""
                        txt = (txt or "").strip()
                        if txt:
                            f.write(json.dumps({"text": txt, "lang": "zh"}, ensure_ascii=False) + "\n")
                            count += 1
                            if count >= args.zh_limit:
                                break
                    except Exception:
                        continue
            zh_written = count
        except Exception:
            zh_written = 0

    # Optional bitext (very small sample for alignment) and fallback for mono
    bitext_written = 0
    if args.bitext or en_written == 0 or zh_written == 0:
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
            # Fallback: populate mono files from bitext if missing
            if en_written == 0:
                with (out_dir / "bitext.tsv").open("r", encoding="utf-8") as f_in, (out_dir / "en.jsonl").open("w", encoding="utf-8") as f_en:
                    for ln_i, ln in enumerate(f_in):
                        en = ln.split("\t")[0].strip()
                        if en:
                            f_en.write(json.dumps({"text": en, "lang": "en"}) + "\n")
                            if ln_i + 1 >= args.en_limit:
                                break
                en_written = min(args.en_limit, bitext_written)
            if zh_written == 0:
                with (out_dir / "bitext.tsv").open("r", encoding="utf-8") as f_in, (out_dir / "zh.jsonl").open("w", encoding="utf-8") as f_zh:
                    for ln_i, ln in enumerate(f_in):
                        parts = ln.rstrip("\n").split("\t")
                        zh = parts[1].strip() if len(parts) > 1 else ""
                        if zh:
                            f_zh.write(json.dumps({"text": zh, "lang": "zh"}) + "\n")
                            if ln_i + 1 >= args.zh_limit:
                                break
                zh_written = min(args.zh_limit, bitext_written)
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
