"""Reserve previously unused plural and singleton questions for whole-set training."""

from scripts.prepare_multiple_ink_runs import (
    ROOT,
    WORK as ORIGINAL,
    prepared_files,
    read,
)
from scripts.reserve_survival_pages import reserve

WORK = ROOT / "artifacts/complete_fact_sets_v1"
PUB = ROOT / "references/evidence/complete_fact_sets_v1"
MODEL = ROOT / "models/complete_fact_sets"


def main():
    prior = read(ORIGINAL / "held.json") + read(ORIGINAL / "confirm.json")
    groups = (("en", True, 64), ("zh", True, 96), ("zh", False, 64))
    prepared_files(
        WORK,
        PUB / "fresh.json",
        {"confirm.json": ("sha256",), "confirm-tiles.npy": ("tiles",)},
        lambda stage: reserve(
            ORIGINAL, stage, prior, groups, "complete-fact-sets-v1 ", 400
        ),
    )


if __name__ == "__main__":
    main()
