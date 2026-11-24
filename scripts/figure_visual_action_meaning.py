"""Export the registered counts and actual emitted Pu Songling strips for Figure 125."""

import numpy as np
from PIL import Image

from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.prepare_visual_action_meaning import STUDY, read, dump
from scripts.train_typeface_writer import FACES, Face


def main():
    base = read(STUDY / "evaluation/result.json")["groups"]
    fresh = read(STUDY / "pu_evaluation/result.json")
    arms = ("parent", "aligned", "exchanged")
    table = [["Two-line pages", "Parent", "Aligned", "Exchanged"]]
    for title, g in [
        ("Original names (1,691)", "literal"),
        ("Tang descriptions (9)", "tang_webster"),
    ]:
        table.append([title, *map(str, base[g]["written"])])
    table.append(
        [
            "Fresh Pu descriptions (10)",
            *[str(fresh["groups"]["webster_" + a]["written"]) for a in arms],
        ]
    )
    dump(
        STUDY / "figure.json",
        dict(
            table=table,
            series=[
                [
                    base[k]["written"][i]
                    for k in ("editorial", "webster", "webster_multiword")
                ]
                for i in range(3)
            ],
        ),
    )
    rows = read(STUDY / "pu_screen.json")["rows"][16:18]
    face = Face(FACES[0])
    page = Image.new("L", (max(len(r["line"]) for r in rows) * 32, 64), 255)
    for i, r in enumerate(rows):
        bits = (
            np.unpackbits(np.frombuffer(face(r["line"]), np.uint8))
            .reshape(-1, 32, 32)
            .transpose(1, 0, 2)
            .reshape(32, -1)
        )
        page.paste(Image.fromarray(255 * (1 - bits)), (0, i * 32))
    page.save(STUDY / "pu-page.png")
    records = read(STUDY / "pu_evaluation/requests.json")
    with np.load(STUDY / "pu_evaluation/emitted.npz") as a:
        for i, r in enumerate(records):
            if (r["arm"], r["mode"], r["pair"], r["reverse"]) == (
                "aligned",
                "webster",
                8,
                False,
            ) and r["question"].startswith("What"):
                write_strip(
                    a["pixels"][a["offsets"][i] : a["offsets"][i + 1]],
                    STUDY / f"pu-answer-{r['side']}.png",
                )


if __name__ == "__main__":
    main()
