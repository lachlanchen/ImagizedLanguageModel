"""Build the edited-page figure using unmodified stopped outputs and paired scores."""

import json
import numpy as np
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_typeface_writer import Face, FACES
from scripts.role_change_corpus import load, target
from scripts.prepare_role_change_actions import WORK, PUBLIC
from scripts.evaluate_pointing_causal_writer import load_streams


def png(raw, path):
    write_strip(np.frombuffer(raw, np.uint8).reshape(-1, 128), path)


def main():
    scores = json.loads((PUBLIC / "scores.json").read_text())
    reading = json.loads((PUBLIC / "reading_result.json").read_text())["scores"]
    table = [["System", "Credit contrasts", "Story contrasts", "Full statements"]]
    series = []
    for a, label in (
        ("previous", "Earlier writer"),
        ("independent", "Independent labels"),
        ("aligned", "Aligned labels"),
        ("qwen", "Text Qwen3-8B"),
    ):
        contrasts = []
        if a != "qwen":
            cats = []
            for g in ("duie", "fresh"):
                group = scores[a][g]
                counts = group["counts"]
                contrasts.append(
                    f"{sum(v['complete'] for v in counts.values())} / {2*group['scenes']}"
                )
                cats.extend(
                    round(100 * counts[lang]["ordered"] / counts[lang]["requests"], 1)
                    for lang in ("zh", "en")
                )
            series.append(cats)
        else:
            contrasts = ["Unmeasured"] * 2
        table.append([label, *contrasts, f"{reading[a]['complete']} / 48"])
    path = WORK / "fresh_aligned.npz"
    assert sha(path) == scores["aligned"]["fresh"]["sha256"]
    ys, ends = load_streams(path)
    face = Face(FACES[0])
    groups = load()
    # The first registered authored story fixes the illustrated page before scoring.
    c = groups["fresh"][0]
    examples = PUBLIC / "examples"
    examples.mkdir(exist_ok=True)
    for i, q in enumerate(c["qs"]["zh"]):
        png(face(q), examples / f"q{i}.png")
    strips = []
    for n, label, img in (
        ("question_a", "First question", face(c["qs"]["zh"][0])),
        ("question_b", "Second question", face(c["qs"]["zh"][1])),
    ):
        png(img, PUBLIC / (n + ".png"))
        strips.append([n, label])
    for w in (0, 1):
        png(face(c["pages"][w]), examples / f"page{w}.png")
        for name, label, s in [
            (f"target{w}", f"Page {w+1} · example answer", face(target(c, w, False))),
            (f"actual{w}", f"Page {w+1} · generated", ys[4 * w]),
        ]:
            png(s, PUBLIC / (name + ".png"))
            strips.append([name, label])
    credit_path = WORK / "duie_aligned.npz"
    assert sha(credit_path) == scores["aligned"]["duie"]["sha256"]
    pixels, _ = load_streams(credit_path)
    checks = json.loads((WORK / "duie_aligned.json").read_text())
    index = next(
        i
        for i in range(len(groups["duie"]))
        if all(v["exact"] for v in checks[i * 8 : i * 8 + 8])
    )
    credit = groups["duie"][index]
    for w in (0, 1):
        name = f"credit{w}"
        expect = target(credit, w, False)
        raw = pixels[8 * index + 4 * w]
        assert raw == face(expect)
        png(raw, PUBLIC / (name + ".png"))
        strips.append([name, f"Credit {w+1} · generated"])
    record(
        PUBLIC / "figure.json",
        dict(
            series=series,
            table=table,
            strips=strips,
            case=c["key"],
            credit_case=credit["key"],
            stops=[ends[0], ends[4]],
        ),
    )


if __name__ == "__main__":
    main()
