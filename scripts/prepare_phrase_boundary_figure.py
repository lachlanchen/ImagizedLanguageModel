"""Export exact source and model strips and their measured numerical comparisons."""

from scripts.prepare_phrase_boundary_trial import EVIDENCE, OUT, load, write_new
from scripts.evaluate_paired_student import read_streams
from scripts.infer_visual_translation import page


def run():
    sources = {
        "source": read_streams("source", OUT),
        "single": read_streams("single", OUT),
        "joint": read_streams("joint", OUT),
    }
    widths = {}
    for i, tag in [
        (12, "camps"),
        (14, "bridge"),
        (1, "depart"),
        (18, "elm"),
        (33, "army"),
    ]:
        for name, seq in sources.items():
            key = tag + "_" + name
            page(seq[i]).save(EVIDENCE / (key + ".png"))
            widths[key] = len(seq[i])
    scores = load(EVIDENCE / "scores.json")["scores"]
    table = [["Writer", "Archive pairs¹", "Authored¹", "Other archive", "Diverse"]]
    for k, label in [
        ("composer", "Phrase composer"),
        ("single", "Single cell"),
        ("joint", "Joint span"),
        ("cue", "Explicit cues"),
    ]:
        vals = scores[k]
        table.append(
            [
                label,
                str(vals["archive_pair"]["both_faithful"]) + " / 6",
                str(vals["authored_pair"]["both_faithful"]) + " / 4",
                str(vals["archive_single"]["faithful"]) + " / 17",
                str(vals["broad"]["faithful"]) + " / 12" if "broad" in vals else "—",
            ]
        )
    reach = load(EVIDENCE / "reach.json")
    write_new(
        EVIDENCE / "figure.json",
        dict(
            table=table,
            widths=widths,
            series=[[reach["exact"], *[reach["counts"][str(k)] for k in (1, 3, 6)]]],
        ),
    )


if __name__ == "__main__":
    run()
