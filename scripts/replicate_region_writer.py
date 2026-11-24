"""Run an independent fit using the released trainer and authenticated optical corpus."""

import argparse
from pathlib import Path
import shutil

from scripts import prepare_region_writer as corpus
from scripts import train_region_writer as trainer
from scripts.verse_gpu_runtime import configure_cuda_cli


def run(directory):
    corpus.main()
    root = directory.resolve()
    model = root / "model"
    work = root / "work"
    if root == corpus.WORK.resolve() or model.resolve() == corpus.MODEL.resolve():
        raise ValueError("Choose a separate replication directory.")
    model.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    for name in corpus.CORPUS_FILES:
        source = (corpus.WORK / name).resolve()
        dest = work / name
        if dest.is_symlink() or dest.exists():
            if dest.resolve() != source:
                raise ValueError(f"Unexpected corpus path {dest}")
        else:
            dest.symlink_to(source)
    palette = model / "palette.npy"
    if palette.exists() and corpus.sha(palette) != corpus.sha(work / palette.name):
        raise ValueError("Changed replication bitmap bank")
    if not palette.exists():
        shutil.copyfile(work / palette.name, palette)
    # The released fitter binds its own source bytes into every checkpoint.
    # Set only the output and resume roots, leaving that fitter byte-identical.
    trainer.MODEL, trainer.WORK = model, work
    for arm in ("selector", "direct", "selected"):
        trainer.fit(arm)


if __name__ == "__main__":
    configure_cuda_cli()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    run(parser.parse_args().directory)
