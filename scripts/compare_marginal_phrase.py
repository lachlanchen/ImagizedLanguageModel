"""Measure the two-stage sentence writer and local Q4 visual baseline on new passages."""

from scripts.compare_region_writer import compare
from scripts.prepare_open_exam import read
from scripts.prepare_marginal_confirmation import WORK, PUB
from scripts.train_marginal_phrase import MODEL
from scripts.verse_gpu_runtime import configure_cuda_cli

if __name__ == "__main__":
    configure_cuda_cli()
    compare(read(WORK / "confirmation.json"), PUB, MODEL)
