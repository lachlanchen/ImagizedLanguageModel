#!/usr/bin/env python3
"""Register local research books without duplicating or committing source binaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "references/source_books/catalog.json"
DEFAULT_LIBRARY = ROOT / "references/source_books/library"
DEFAULT_MANIFEST = ROOT / "references/source_books/manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_metadata(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["pdfinfo", str(path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}

    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip().lower().replace(" ", "_")] = value.strip()
    metadata: dict[str, Any] = {}
    if values.get("pages", "").isdigit():
        metadata["pages"] = int(values["pages"])
    for source_key, output_key in (
        ("page_size", "page_size"),
        ("pdf_version", "pdf_version"),
        ("encrypted", "encrypted"),
    ):
        if source_key in values:
            metadata[output_key] = values[source_key]
    return metadata


def media_type(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "application/pdf"
    if path.suffix.lower() == ".epub":
        return "application/epub+zip"
    return "application/octet-stream"


def ensure_link(source: Path, link: Path) -> None:
    expected = Path(os.path.relpath(source, link.parent))
    if link.is_symlink():
        if Path(os.readlink(link)) != expected:
            raise RuntimeError(f"nonmatching symlink exists: {link}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace local file: {link}")
    link.symlink_to(expected)


def build_manifest(
    catalog: dict[str, Any],
    source_dir: Path,
    library_dir: Path,
) -> dict[str, Any]:
    registered: list[dict[str, Any]] = []
    library_dir.mkdir(parents=True, exist_ok=True)
    for item in catalog["items"]:
        source = (source_dir / item["filename"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        link = library_dir / item["filename"]
        ensure_link(source, link)
        entry = {
            **item,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "media_type": media_type(source),
            "source_path": str(source),
            "local_link": str(link.relative_to(ROOT)),
            "rights_status": "unverified-local-research-only",
        }
        if source.suffix.lower() == ".pdf":
            entry["pdf"] = pdf_metadata(source)
        registered.append(entry)
    return {
        "schema_version": catalog["schema_version"],
        "source_root": str(source_dir.resolve()),
        "library_root": str(library_dir.relative_to(ROOT)),
        "rights_policy": catalog["rights_policy"],
        "total_bytes": sum(item["bytes"] for item in registered),
        "items": registered,
    }


def verify_manifest(manifest: dict[str, Any]) -> None:
    errors: list[str] = []
    for item in manifest["items"]:
        source = Path(item["source_path"])
        link = ROOT / item["local_link"]
        if not source.is_file():
            errors.append(f"missing source: {source}")
            continue
        if not link.is_symlink() or link.resolve() != source.resolve():
            errors.append(f"invalid local link: {link}")
        if source.stat().st_size != item["bytes"]:
            errors.append(f"size changed: {source}")
            continue
        digest = sha256_file(source)
        if digest != item["sha256"]:
            errors.append(f"hash changed: {source}")
    if errors:
        raise RuntimeError("reference registry verification failed:\n" + "\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--library-dir", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        verify_manifest(manifest)
        print(
            f"verified={len(manifest['items'])} bytes={manifest['total_bytes']} "
            f"manifest={args.manifest}"
        )
        return 0

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    manifest = build_manifest(catalog, args.source_dir, args.library_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"registered={len(manifest['items'])} bytes={manifest['total_bytes']} "
        f"manifest={args.manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
