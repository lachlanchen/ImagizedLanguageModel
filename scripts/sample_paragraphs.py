#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main():
    ap = argparse.ArgumentParser(description="Sample mixed EN/ZH paragraphs into a small test set")
    ap.add_argument("--en", default="data/raw/en.jsonl", help="path to EN jsonl")
    ap.add_argument("--zh", default="data/raw/zh.jsonl", help="path to ZH jsonl")
    ap.add_argument("--out", default="data/processed/test_100.jsonl", help="output jsonl")
    ap.add_argument("--n_total", type=int, default=100, help="total paragraphs to sample")
    ap.add_argument("--ratio_en", type=float, default=0.5, help="portion of EN in mix [0..1]")
    args = ap.parse_args()

    n_en = int(args.n_total * args.ratio_en)
    n_zh = args.n_total - n_en

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    cnt_en = 0
    cnt_zh = 0

    with open(outp, "w", encoding="utf-8") as f:
        # EN
        try:
            for ex in stream_jsonl(args.en):
                txt = (ex.get("text") or "").strip()
                if not txt:
                    continue
                f.write(json.dumps({"text": txt, "lang": "en"}, ensure_ascii=False) + "\n")
                cnt_en += 1
                if cnt_en >= n_en:
                    break
        except FileNotFoundError:
            pass
        # ZH
        try:
            for ex in stream_jsonl(args.zh):
                txt = (ex.get("text") or "").strip()
                if not txt:
                    continue
                f.write(json.dumps({"text": txt, "lang": "zh"}, ensure_ascii=False) + "\n")
                cnt_zh += 1
                if cnt_zh >= n_zh:
                    break
        except FileNotFoundError:
            pass

    print(json.dumps({"wrote": cnt_en + cnt_zh, "en": cnt_en, "zh": cnt_zh}))


if __name__ == "__main__":
    main()

