"""Apply the existing blinded original-and-edit semantic calibration to the new cohort."""

from scripts.prepare_fact_channel import WORK, PUB
from scripts.review_phrase_region import review
from scripts.verse_gpu_runtime import configure_cuda_cli


if __name__ == "__main__":
    configure_cuda_cli()
    files = {"direct": "direct", "fact_only": "human", "fact_region": "predicted"}
    review(WORK, PUB, "fresh", tuple(files), files=files)
