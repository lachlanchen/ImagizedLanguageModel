"""Render a pair of freely drawn story answers and the measured complete-answer objectives."""

from hashlib import sha256
import matplotlib.pyplot as plt
import numpy as np
from scripts.evaluate_story_answer_fit import material, ARMS
from scripts.prepare_story_answer_fit import WORK, PUB, MODEL, read, dump
from scripts.prepare_fact_channel import MODEL as BASE, Ink
from scripts.evaluate_directional_reading import bitmap
from scripts.evaluate_visual_discourse import unpack
from scripts.figure_directional_reading import save_strip
from scripts.figure_recurrent_evidence import Plate, TEAL, MUTED, FIG
from scripts.figure_phrase_region import breaks


def images():
    rows = material()
    out = {m: unpack(WORK / (m + ".json.gz"))["original"] for m in ARMS}
    judged = read(PUB / "semantic.json")["votes"]
    ids = read(PUB / "comparison.json")["indices"]
    eligible = []
    for j, i in enumerate(ids):
        if rows[i]["task"] != "stories":
            continue
        if any(
            max(
                len(rows[k]["q"]),
                len(rows[k]["answers"][0]),
                *[out[m][k]["cells"] for m in ARMS],
            )
            > 135
            for k in (i, i ^ 1)
        ):
            continue
        if any(not out[m][k]["stop"] for k in (i, i ^ 1) for m in ARMS):
            continue
        changed = out["marginal"][i]["sha256"] != out["signed"][i]["sha256"]
        credit = lambda m: all(judged[2 * j + d][m] for d in range(2))
        eligible.append(
            (
                not (changed and credit("marginal") != credit("signed")),
                not changed,
                j,
                i,
            )
        )
    _, _, _, pick = min(eligible)
    assets = PUB / "plates"
    assets.mkdir(exist_ok=True)
    ink = Ink(np.load(BASE / "palette.npy"))
    plate = Plate("Figure_182_learning_from_whole_replies", assets)
    examples = []
    for col, i in enumerate((pick, pick ^ 1)):
        r = rows[i]
        folder = MODEL / "examples" / ("story-" + str(col + 1))
        folder.mkdir(parents=True, exist_ok=True)
        q = save_strip(folder, "question", bitmap(ink, r["q"]))
        blocks = [
            save_strip(folder, f"region-{k}", bitmap(ink, b))
            for k, b in enumerate(r["blocks"])
        ]
        dump(folder / "input.json", dict(question=q, regions=blocks))
        fields = [
            ("Question", q, r["q"]),
            (
                "Human reference",
                save_strip(folder, "human", bitmap(ink, r["answers"][0])),
                r["answers"][0],
            ),
        ]
        for m, label in zip(ARMS, ("Parent", "Single path", "Marginal"), strict=True):
            a = out[m][i]
            b = bytes.fromhex(a["ink_hex"])
            assert sha256(b).hexdigest() == a["sha256"]
            fields.append((label, save_strip(folder, m, b), a["text"]))
        y = col * 135
        plate.label(5, y + 2, "ab"[col], 12, True)
        plate.label(
            24,
            y + 2,
            ("A question on an unseen story", "A second question on the same section")[
                col
            ],
            10,
            True,
        )
        lines = [breaks(v) for _, _, v in fields]
        dy = min(18, 99 / sum(len(e) - 1 for e in lines))
        y += 19
        for (label, spec, _), edges in zip(fields, lines, strict=True):
            plate.label(6, y, label, 8, True, TEAL if label == "Marginal" else MUTED)
            for line, (a, b) in enumerate(zip(edges, edges[1:])):
                if b > a:
                    plate.strip(
                        f"{col}-{spec['png']}-{line}",
                        folder / spec["png"],
                        spec["cells"],
                        95,
                        y + dy * line,
                        441,
                        dy,
                        a,
                        b,
                    )
            y += (len(edges) - 1) * dy + 2
        examples.append(
            dict(
                i=i,
                question=r["q"],
                answers=r["answers"],
                outputs={
                    m: {
                        k: out[m][i][k]
                        for k in ("text", "fact_sha256", "sha256", "stop")
                    }
                    for m in ARMS
                },
            )
        )
    plate.rule(132)
    dump(PUB / "plates.json", [plate.render()])
    dump(PUB / "examples.json", examples)


def plot():
    curves = {
        m: [r["mean_joint_nll"] for r in read(MODEL / m / "fit.json")["epochs"]]
        for m in ARMS[1:]
    }
    sem = read(PUB / "semantic.json")["summary"]
    dump(PUB / "trajectory.json", curves)
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(7.6, 2.75), gridspec_kw={"width_ratios": [1, 1.3]}
    )
    for m, c, label in zip(
        ARMS[1:], ("#547ba1", TEAL), ("Single path", "Marginal"), strict=True
    ):
        a.plot(
            range(1, 9),
            curves[m],
            label=label,
            color=c,
            marker="o" if m == "marginal" else "s",
            ms=3,
        )
    a.set(
        xlabel="Training pass",
        ylabel="Mean full-answer NLL (nats)",
        xticks=[1, 2, 4, 6, 8],
    )
    tasks = ("stories", "exam", "mlqa-en", "mlqa-zh")
    x = np.arange(4)
    for k, (m, name, c) in enumerate(
        zip(
            (*ARMS, "qwen"),
            ("Parent", "Single path", "Marginal", "Qwen"),
            ("#928377", "#547ba1", TEAL, "#c6a257"),
            strict=True,
        )
    ):
        b.bar(
            x + (k - 1.5) * 0.21, [sem[t][m] for t in tasks], 0.2, color=c, label=name
        )
    b.set(
        xticks=x,
        xticklabels=["New story", "Exam", "MLQA EN", "MLQA ZH"],
        ylabel="Semantic credits / 16",
        ylim=(0, 17),
    )
    for ax, title in (
        (a, "a   Learning complete replies"),
        (b, "b   Free answers from images"),
    ):
        ax.set_title(title, fontweight="bold", loc="left", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=6.7, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("Figure_183_full_answer_learning." + ext), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    images()
    plot()
