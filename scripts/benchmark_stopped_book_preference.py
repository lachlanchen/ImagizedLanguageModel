"""Compare the preference writer and text Qwen on the newly reserved book requests."""

from scripts.benchmark_first_glyph_books import run
from scripts.mine_stopped_book_replies import WORK, PUBLIC
from scripts.reserve_preference_books import PAGES, literary_cases
from scripts.train_span_choice import gpu_owner

if __name__ == "__main__":
    with gpu_owner():
        run(WORK, PUBLIC, PAGES, literary_cases, arms=("preference", "replay"))
