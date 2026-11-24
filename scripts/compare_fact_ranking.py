"""Apply the matched local 8B vision and blinded semantic comparisons."""

from scripts.prepare_fact_ranking_confirmation import PUB, WORK
from scripts.benchmark_fact_channel import main as benchmark
from scripts.review_phrase_region import review
from scripts.write_fact_ranking import RankedAnswer
from scripts.verse_gpu_runtime import configure_cuda_cli


if __name__ == "__main__":
    configure_cuda_cli()
    modes = ("frozen", "endpoints", "whole")
    benchmark(WORK, PUB, {m: lambda mode=m: RankedAnswer(mode, "cuda") for m in modes})
    review(WORK, PUB, "", modes)
