"""Replay the wider writer and reproduce the originally stored PNGs in print."""

import torch
from scripts.figure_book_rehearsal import plate
from scripts.figure_visual_prefix import ASSETS
from scripts.prepare_visual_rollins import PUB, read, sha
from scripts.reserve_wide_translation import WORK, PUB as WIDTH
from scripts.write_ordered_translation import restore, infer
from ilm.visual_lm.portable_writer import read_image, save_grid


def main():
    torch.set_num_threads(2)
    gen = read(WORK / "generated.json")
    grades = read(WIDTH / "results.json")
    n = read(WIDTH / "reservation.json")["directions"]
    assert sha(WORK / "generated.json") == grades["generated"]
    # Show the first article in both directions. If the model earns a fresh
    # credit, include the first one in the original order.
    cases = [
        (0, "a    First reserved article, Chinese to English"),
        (1, "b    Same article, English to Chinese"),
    ]
    wins = [
        i
        for i in range(n)
        if all(grades["votes"][2 * i + k]["wide"] for k in (0, 1))
        and gen["answers"][i]["wide"]["stop"]
    ]
    if wins and wins[0] not in (0, 1):
        cases.append(
            (wins[0], "c    First credited request from the reserved articles")
        )
    else:
        cases.append((n + 145, "c    First held passive construction"))
    groups = []
    networks = {
        m: restore(tag, "cpu")
        for m, tag in [("small", "prefix-reference"), ("wide", "wide")]
    }
    for i, heading in cases:
        r = gen["rows"][i]
        folder = WORK / "pixels" / f"{i:04d}"
        input_path = folder / "input.png"
        actual = read_image(input_path, len(r["input"]))
        shown = [("Pictured input", f"width-{i}-input", input_path)]
        for m, model in networks.items():
            with torch.inference_mode():
                raw, stop = infer(model, actual, r["t"], "cpu")
            p = WORK / (f"replay-{i}-{m}.png")
            save_grid(raw, p)
            expected = gen["answers"][i][m]
            assert stop == expected["stop"] and sha(p) == expected["png"] == sha(
                folder / (m + ".png")
            )
            shown.append(
                (
                    {"small": "1.69M writer", "wide": "14.90M writer"}[m],
                    f"width-{i}-{m}",
                    folder / (m + ".png"),
                )
            )
        groups.append((heading, shown))
    plate(
        "width-rasters",
        "Figure_203_wider_visual_writing",
        groups,
        assets=ASSETS,
        evidence=PUB,
    )


if __name__ == "__main__":
    main()
