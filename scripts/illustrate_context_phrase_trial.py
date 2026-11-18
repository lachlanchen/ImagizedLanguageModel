"""Build Figure 69 from the two directions of the first registered role swap."""

import numpy as np
from PIL import Image
from scripts.evaluate_paired_student import read_streams
from scripts.reserve_context_phrase_trial import EVIDENCE, load, write_new
from scripts.train_context_phrase_trial import sha


def run():
    result = load(EVIDENCE / "scores.json")
    table = [["Method", "Both training choices / 39", "Both new choices / 15", "Faithful / 18", "Both details / 15"]]
    labels = dict(parent="Starting writer", rehearsal="Rehearsal", crossed="Crossed phrases",
                  ink_overlap="Source ink overlap", composer="Phrase composer", source="Literal copy",
                  teacher="Qwen3-8B (text)")
    for method, label in labels.items():
        forced = result["forced"].get(method)
        free = result["free"].get(method)
        table.append([label, *[str(len(forced[k]["strict"])) if forced else "—" for k in ("train","test")],
                      *[str(len(free[k])) if free else "—" for k in ("faithful","both_detail")]])
    folder = EVIDENCE / "streams"
    pairs = load(EVIDENCE / "evaluation.json")["pair_rows"]
    # Show a credited single answer beside the first prespecified role reversal.
    selected = [*pairs[6], pairs[8][0]]
    strips = {}
    for method in ("source", "crossed"):
        all_rows = read_streams(method, folder)
        for i, row in enumerate(selected):
            strip = all_rows[row]
            pixels = np.unpackbits(strip, axis=1).reshape(-1, 32, 32).transpose(1, 0, 2).reshape(32, -1)
            tag = method + str(i)
            path = EVIDENCE / (tag + ".png")
            Image.fromarray(255 * (1 - pixels)).save(path)
            strips[tag] = dict(row=row, cells=len(strip), png_sha=sha(path),
                               source_sha=sha(folder / (method + ".npz")))
    series = [[round(line[2], 4) for line in load(EVIDENCE / (arm + "_fit.json"))["losses_by_97"]]
              for arm in ("rehearsal", "crossed")]
    write_new(EVIDENCE / "figure.json", dict(strips=strips, table=table, series=series))


if __name__ == "__main__":
    run()
