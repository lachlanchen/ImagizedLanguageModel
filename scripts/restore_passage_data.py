"""Reconstruct the frozen passage splits from their published books and hashes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
import os

from scripts import prepare_passage_selector as prep
from scripts.evaluate_book_page_memory import HASH, URL
from scripts.measure_book_discourse import DIGEST, URL as TEST_URL


def restore(verify_rebuild=False):
    data = prep.base.read(prep.PUB / "data.json")
    prep.WORK.mkdir(parents=True, exist_ok=True)
    files = data["files"]
    present = [k for k in files if (prep.WORK / k).exists()]
    for k in present:
        if prep.base.sha(prep.WORK / k) != files[k]:
            raise ValueError(f"Local split differs: {k}")
    if len(present) == len(files) and not verify_rebuild:
        print("Verified all five working data files")
        return
    prep.download(URL, prep.RAW / "lambada-validation.parquet", HASH)
    prep.download(TEST_URL, prep.RAW / "lambada-test.jsonl", DIGEST)
    old_books = prep.base.read(
        prep.ROOT / "references/evidence/book_order_prior_v1/data.json"
    )["books"]["train"]
    test_books = [
        {k: b[k] for k in ("name", "generation", "sha")}
        for b in data["books"]["gutenberg"]
    ]

    def fetch(b):
        prep.download(
            prep.GCS + b["name"] + "?generation=" + b["generation"],
            prep.RAW / b["name"],
            b.get("sha") or b["sha256"],
        )

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(fetch, [*old_books, *test_books]))
    with TemporaryDirectory(dir=prep.WORK.parent, prefix="passage-restore-") as temp:
        root = Path(temp)
        spec = spec_from_file_location("frozen_passage_preparation", prep.__file__)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        module.WORK = root / "splits"
        module.MODEL = root / "model"
        module.PUB = root / "evidence"
        module.fresh_pg = lambda: test_books
        module.main()
        # Set iteration order changes JSON key order in the old document-frequency file.
        # The frozen distribution must agree at every word, whatever the local hash seed.
        assert prep.base.read(
            module.MODEL / "document_frequency.json"
        ) == prep.base.read(prep.MODEL / "document_frequency.json")
        rebuilt = prep.base.read(module.PUB / "data.json")
        assert {k: v for k, v in data.items() if k != "prior_df"} == {
            k: v for k, v in rebuilt.items() if k != "prior_df"
        }
        for k, expected in files.items():
            if prep.base.sha(module.WORK / k) != expected:
                raise ValueError(f"Failed reconstruction of {k}")
        for k in files.keys() - set(present):
            os.replace(module.WORK / k, prep.WORK / k)
    print("Reconstructed and independently verified all five working data files")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verify-rebuild", action="store_true")
    restore(cli.parse_args().verify_rebuild)
