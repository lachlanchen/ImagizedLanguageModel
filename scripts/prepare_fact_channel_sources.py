"""Restore the pinned Chinese reading sources needed by the fact-channel corpus."""

from pathlib import Path
from scripts.prepare_fact_channel import ROOT
from scripts.prepare_visual_cursor import FILES
from scripts.prepare_visual_discourse import fetch
from scripts.prepare_visual_reading_qa import BASE, SOURCES
from scripts.prepare_region_writer import ensure_raw_sources


def main():
    for name, h in FILES["cmrc"]:
        fetch(ROOT / "data/raw" / name, h, [BASE + SOURCES[Path(name).name][0]])
    name, h = FILES["drcd"][0]
    fetch(
        ROOT / "data/raw" / name,
        h,
        [
            "https://raw.githubusercontent.com/DRCKnowledgeTeam/DRCD/b944790de5af02c5fbb7cd9cb1473d27d169eebf/"
            + Path(name).name
        ],
    )
    ensure_raw_sources()


if __name__ == "__main__":
    main()
