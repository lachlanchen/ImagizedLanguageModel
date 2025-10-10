#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from urllib.request import urlopen


EN_CANDIDATES = [
    "https://raw.githubusercontent.com/tatsu-lab/alpaca/main/alpaca_data.json",
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
    "https://raw.githubusercontent.com/yahma/alpaca-cleaned/main/alpaca_data_cleaned.json",
]

ZH_CANDIDATES = [
    "https://raw.githubusercontent.com/shibing624/alpaca-chinese-dataset/main/alpaca_chinese_dataset.json",
    "https://raw.githubusercontent.com/ymcui/Chinese-LLaMA-Alpaca/main/data/alpaca_chinese_dataset.json",
    "https://raw.githubusercontent.com/LC1332/Chinese-alpaca-lora/main/data/alpaca_chinese_dataset.json",
]


def try_download(url: str) -> str:
    try:
        with urlopen(url, timeout=30) as resp:
            if resp.status != 200:
                return ""
            data = resp.read().decode("utf-8", errors="ignore")
            return data
    except Exception:
        return ""


def validate_alpaca_json(text: str) -> bool:
    try:
        obj = json.loads(text)
        if isinstance(obj, list) and obj:
            sample = obj[0]
        elif isinstance(obj, dict) and "data" in obj:
            sample = obj.get("data") or []
            sample = sample[0] if sample else {}
        else:
            return False
        # Look for typical keys
        keys = set(sample.keys())
        expect = {"instruction", "input", "output"}
        return len(keys.intersection(expect)) >= 2
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Download Alpaca finetune datasets (EN and ZH) from GitHub")
    ap.add_argument("--outdir", default="data/raw", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {"en": None, "zh": None}

    # English
    for url in EN_CANDIDATES:
        txt = try_download(url)
        if txt and validate_alpaca_json(txt):
            (outdir / "alpaca_en.json").write_text(txt, encoding="utf-8")
            results["en"] = url
            break

    # Chinese
    for url in ZH_CANDIDATES:
        txt = try_download(url)
        if txt and validate_alpaca_json(txt):
            (outdir / "alpaca_zh.json").write_text(txt, encoding="utf-8")
            results["zh"] = url
            break

    print(json.dumps({
        "saved_en": str(outdir / "alpaca_en.json") if results["en"] else None,
        "source_en": results["en"],
        "saved_zh": str(outdir / "alpaca_zh.json") if results["zh"] else None,
        "source_zh": results["zh"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

