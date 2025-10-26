#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import shutil
import subprocess


DEFAULT_MANIFEST = {
    # Public, small-ish resources we can mirror quickly (examples)
    "examples": [
        # Add small demo sets if available
    ],
    # Large academic datasets (manual or gated; set url to actual archive when available)
    "datasets": [
        {
            "name": "HUST-OBC (Oracle Bone Characters)",
            "id": "hust_obc",
            "url": None,  # TODO: insert official archive URL (zenodo/figshare/other)
            "notes": "Large dataset (~140k imgs). Some mirrors require registration; add direct URL here to auto-download.",
            "dest": "data/historic/hust_obc",
        },
        {
            "name": "Tangut Character Dataset (TCD/TCD-E)",
            "id": "tangut_tcd",
            "url": None,  # TODO
            "notes": "~6k classes; 120k+ images. Provide direct archive URL if permitted.",
            "dest": "data/historic/tangut_tcd",
        },
        {
            "name": "NomNaOCR Han-Nom (demo subset)",
            "id": "nomnaocr_demo",
            "url": "https://raw.githubusercontent.com/ds4v/NomNaOCR/main/README.md",
            "notes": "Demo: pulls README only to verify connectivity. Replace with dataset archive if license permits.",
            "dest": "data/historic/nomnaocr_demo",
        },
    ]
}


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_url(url: str, dest_dir: Path) -> None:
    ensure_dir(dest_dir)
    # Use curl or wget if available
    target = dest_dir / os.path.basename(url)
    if shutil.which("curl"):
        run(["curl", "-L", "-o", str(target), url])
    elif shutil.which("wget"):
        run(["wget", "-O", str(target), url])
    else:
        raise SystemExit("Neither curl nor wget found.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download historical script datasets into data/historic")
    ap.add_argument("--manifest", default=None, help="JSON manifest with dataset entries; defaults to built-in")
    ap.add_argument("--out-root", default="data/historic", help="Root folder for downloads")
    ap.add_argument("--only", nargs="*", default=None, help="IDs to download (subset)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    manifest = DEFAULT_MANIFEST
    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    selected = set(args.only or [])
    print("Download root:", out_root)

    # Try example small resources first (connectivity check)
    for entry in manifest.get("examples", []):
        url = entry.get("url")
        dest = Path(entry.get("dest", out_root))
        if not url:
            continue
        try:
            print(f"[example] {entry.get('name')}: {url}")
            download_url(url, dest)
        except Exception as e:
            print(f"WARN: failed example download: {e}")

    # Main datasets
    for entry in manifest.get("datasets", []):
        ds_id = entry.get("id")
        if selected and ds_id not in selected:
            continue
        name = entry.get("name")
        url = entry.get("url")
        dest = Path(entry.get("dest", out_root))
        notes = entry.get("notes", "")
        print(f"[dataset] {name}")
        print("  id:", ds_id)
        print("  dest:", dest)
        if not url:
            print("  url: (not set)")
            if notes:
                print("  notes:", notes)
            print("  ACTION: Add a direct archive URL to the manifest and re-run.")
            continue
        try:
            download_url(url, dest)
        except Exception as e:
            print(f"ERROR downloading {name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()

