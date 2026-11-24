"""Render released historical-form replies, new reading samples and measured scores."""

from hashlib import sha256
import sqlite3
from ilm.visual_lm.catalogue_ink import print_cell
import numpy as np
import torch
import matplotlib.pyplot as plt

from scripts.prepare_visual_story_writer import (
    ROOT,
    MODEL,
    PUB,
    WORK,
    OUT,
    STAGES,
    dump,
    read,
)
from scripts.train_balanced_visual_story import PUB as BAL
from scripts.prepare_fact_channel import Ink
from scripts.write_visual_story import StoryWriterAnswer
from scripts.figure_directional_reading import save_strip
from scripts.figure_recurrent_evidence import Plate, FIG, TEAL, MUTED
from scripts.evaluate_visual_story_writer import material, normalize
from scripts.evaluate_visual_discourse import unpack

ASSETS = PUB / "plates"


def catalogue():
    """Four labelled rows on one page, two languages and an open explanation query."""
    folder = MODEL / "examples/yan"
    folder.mkdir(parents=True, exist_ok=True)
    models = {m: StoryWriterAnswer(m, "cpu") for m in ("page", "balanced")}
    ink = Ink(models["page"].palette)
    labels = {}
    blocks = [ink.encode("言。")]
    with sqlite3.connect(OUT / "origin.sqlite3") as db:
        for stage, (zh, en) in STAGES.items():
            label, raw, vector, split = db.execute(
                """SELECT label,pixels32,svg_sha,split FROM form JOIN image USING(svg_sha)
                JOIN entry USING(headword,entry_id) WHERE headword='言' AND stage=? AND has_ink=1
                ORDER BY entry_id,label LIMIT 1""",
                (stage,),
            ).fetchone()
            raw = print_cell(raw)
            if raw not in ink.byink:
                ink.byink[raw] = len(ink.tiles)
                ink.tiles.append(raw)
            idx = ink.byink[raw]
            labels[stage] = dict(label=label, svg_sha=vector, id=idx, split=split)
            blocks.append(ink.encode(f"{zh}（{en}）") + [idx])

    def save(key, ids):
        return save_strip(folder, key, b"".join(ink.tiles[i] for i in ids))

    page = [save("region-" + str(i), b) for i, b in enumerate(blocks)]
    queries = [
        (
            s,
            lang,
            f"Show the {en} form of 言." if lang == "en" else f"请写出言的{zh}字形。",
        )
        for s, (zh, en) in STAGES.items()
        for lang in ("zh", "en")
    ]
    queries.append(
        (
            "explain",
            "en",
            "Explain the evolution of the Chinese character 言 using the evidence on this page.",
        )
    )
    outputs = {}
    for stage, lang, q in queries:
        key = stage + "-" + lang
        ids = ink.encode(q)
        spec = save(key + "-q", ids)
        dump(folder / (key + ".json"), dict(question=spec, regions=page))
        outputs[key] = dict(question=q, models={})
        for m, net in models.items():
            answer, _ = net.answer(ink.array(), [ids], [blocks])
            generated, stop = answer[0]
            raw = b"".join(ink.tiles[k] for k in generated)
            outputs[key]["models"][m] = dict(
                image=save(m + "-" + key, generated),
                stop=stop,
                sha256=sha256(raw).hexdigest(),
                exact=bool(
                    stop and stage in labels and generated == [labels[stage]["id"]]
                )
                if stage in labels
                else None,
            )
    plate = Plate("Figure_184_visual_form_answers", ASSETS)
    plate.label(5, 5, "a", 13, True)
    plate.label(25, 6, "Four attested forms of the character for speech", 11, True)
    for j, (stage, rec) in enumerate(labels.items()):
        x = 107 + 108 * j
        plate.label(x, 32, STAGES[stage][1].capitalize(), 9, True)
        plate.label(x, 46, rec["label"], 8, False, MUTED)
        ref = save("ref-" + stage, [rec["id"]])
        plate.strip("ref-" + stage, folder / ref["png"], 1, x + 8, 59, 58, 37)
        for k, m in enumerate(models):
            spec = outputs[stage + "-zh"]["models"][m]["image"]
            if spec["cells"]:
                plate.strip(
                    m + "-" + stage,
                    folder / spec["png"],
                    spec["cells"],
                    x + 8,
                    109 + 48 * k,
                    75,
                    38,
                )
    for y, name in [(67, "Source"), (117, "Original writer"), (165, "Balanced writer")]:
        plate.label(6, y, name, 8, True, TEAL if y == 165 else MUTED)
    plate.rule(204)
    plate.label(5, 216, "b", 13, True)
    plate.label(
        25, 218, "Question supplied to the model for the first output column", 9, True
    )
    spec = read(folder / "oracle-zh.json")["question"]
    plate.strip("question", folder / spec["png"], spec["cells"], 27, 237, 480, 23)
    dump(
        PUB / "yan.json",
        dict(
            catalogue={
                k: {a: b for a, b in v.items() if a != "id"} for k, v in labels.items()
            },
            outputs=outputs,
        ),
    )
    return plate.render()


def modern():
    rows = material()
    outs = unpack(WORK / "page-replies.json.gz")
    ink = Ink(np.load(ROOT / "models/fact_channel/palette.npy"))
    plate = Plate("Figure_185_reading_new_pages", ASSETS)
    receipt = []
    for j, task in enumerate(("mctest", "c3")):
        eligible = [
            i
            for i, r in enumerate(rows)
            if r["task"] == task
            and outs[i]["stop"]
            and normalize(outs[i]["text"]) in {normalize(a) for a in r["answers"]}
        ]
        i = eligible[0]
        for side, k in enumerate((i, i ^ 1)):
            r = rows[k]
            v = outs[k]
            folder = MODEL / "examples" / (task + str(side))
            folder.mkdir(parents=True, exist_ok=True)

            def typed(name, s):
                return save_strip(
                    folder, name, b"".join(ink.tiles[t] for t in ink.encode(s))
                )

            q = typed("question", r["q"])
            dump(
                folder / "input.json",
                dict(
                    question=q,
                    regions=[
                        typed(f"region-{n}", s) for n, s in enumerate(r["blocks"])
                    ],
                ),
            )
            raw = bytes.fromhex(v["ink_hex"])
            assert sha256(raw).hexdigest() == v["sha256"]
            answer = save_strip(folder, "answer", raw)
            ref = typed("reference", r["answers"][0])
            y = 6 + j * 132 + side * 59
            plate.label(5, y, "abcd"[j * 2 + side], 11, True)
            plate.label(
                22,
                y,
                ("English" if j == 0 else "Chinese")
                + (
                    "  ·  exact answer"
                    if not side
                    else "  ·  second question on the same page"
                ),
                9,
                True,
            )
            for n, (label, spec) in enumerate(
                [("Question", q), ("Reference", ref), ("Output", answer)]
            ):
                z = y + 15 + n * 13
                plate.label(8, z, label, 7, True, TEAL if n == 2 else MUTED)
                if spec["cells"]:
                    plate.strip(
                        f"{j}-{side}-{n}",
                        folder / spec["png"],
                        spec["cells"],
                        68,
                        z,
                        468,
                        12,
                    )
            receipt.append(
                dict(
                    index=k,
                    task=task,
                    question=r["q"],
                    reference=r["answers"][0],
                    text=v["text"],
                    sha256=v["sha256"],
                    stopped=v["stop"],
                )
            )
    plate.rule(129)
    dump(PUB / "reading_examples.json", receipt)
    return plate.render()


def chart():
    counts = read(BAL / "forms.json")
    sem = read(PUB / "semantic.json")["summary"]
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(7.56, 2.65), gridspec_kw={"width_ratios": [0.93, 1.35]}
    )
    labels = ["Page", "Fixed", "Balanced"]
    scores = [counts[k]["all_16_per_family"] for k in ("page", "fixed", "balanced")]
    bars = a.bar(labels, scores, color=["#899792", "#507795", TEAL], width=0.6)
    a.bar_label(bars, padding=1, fontsize=8)
    a.set(
        ylim=(0, 70),
        yticks=[0, 16, 32, 48, 64],
        ylabel="Families passing all 16 requests / 64",
    )
    tasks = ["mctest", "c3", "stories", "exam"]
    for i, (m, c) in enumerate(
        [
            ("parent", "#a99887"),
            ("strip", "#7896aa"),
            ("page", TEAL),
            ("minimind", "#b38838"),
        ]
    ):
        b.bar(
            np.arange(4) + (i - 1.5) * 0.19,
            [sem[t][m] for t in tasks],
            0.18,
            label=m.capitalize(),
            color=c,
        )
    b.set(
        ylim=(0, 17),
        xticks=np.arange(4),
        xticklabels=["MCTest", "C³", "Story", "Exam"],
        ylabel="Replies credited in both orders / 16",
    )
    b.legend(frameon=False, ncol=2, fontsize=6)
    for ax, title in [
        (a, "a   Historical form selection"),
        (b, "b   Meaning of free answers"),
    ]:
        ax.set_title(title, fontsize=10, loc="left", weight="bold")
        ax.tick_params(labelsize=8)
        ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("Figure_186_sealed_forms_and_language." + ext), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    dump(PUB / "plates.json", [catalogue(), modern()])
    chart()
