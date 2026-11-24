"""Run previous optical readers and a local quantized vision comparator on fresh pages."""

from scripts.compare_region_writer import compare
from scripts.prepare_open_exam import read
from scripts.prepare_phrase_confirmation import WORK, PUB
from scripts.train_phrase_region import MODEL
from scripts.verse_gpu_runtime import configure_cuda_cli

if __name__ == "__main__":
    configure_cuda_cli()
    compare(read(WORK / "confirmation.json"), PUB, MODEL)
