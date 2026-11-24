"""Render truthful side-by-side answers from the two candidate scorers."""

from hashlib import sha256
import numpy as np
import matplotlib.pyplot as plt
from scripts.evaluate_optical_competition import ARMS, TASKS
from scripts.evaluate_directional_reading import bitmap
from scripts.evaluate_visual_discourse import unpack
from scripts.figure_directional_reading import save_strip
from scripts.figure_recurrent_evidence import Plate, TEAL, MUTED, FIG
from scripts.figure_phrase_region import breaks
from scripts.prepare_optical_competition import MODEL, PUB, WORK, read, dump
from scripts.prepare_fact_channel import MODEL as BASE, Ink


def make():
    rows = read(WORK / "confirmation.json")
    ink = Ink(np.load(BASE / "palette.npy"))
    runs = {m: unpack(WORK / (m + ".json.gz"))["original"] for m in ARMS}
    assets = PUB / "plates"
    assets.mkdir(exist_ok=True)
    plate = Plate("Figure_179_alternative_visual_answers", assets)
    chosen = []
    for panel, (task, title) in enumerate(
        (
            ("stories", "Two readings of a story"),
            ("exam", "Selecting the answer to a Chinese question"),
        )
    ):
        pool = [
            i
            for i, r in enumerate(rows)
            if r["task"] == task
            and all(runs[m][i]["stop"] for m in ARMS)
            and max(
                len(r["q"]), len(r["answers"][0]), *[runs[m][i]["cells"] for m in ARMS]
            )
            <= 115
        ]

        def exact(i, m):
            return any(
                bytes.fromhex(runs[m][i]["ink_hex"]) == bitmap(ink, v)
                for v in rows[i]["answers"]
            )

        different = [
            i
            for i in pool
            if runs["independent"][i]["sha256"] != runs["competing"][i]["sha256"]
        ]
        gains = [
            i
            for i in different
            if exact(i, "competing") and not exact(i, "independent")
        ]
        losses = [
            i
            for i in different
            if exact(i, "independent") and not exact(i, "competing")
        ]
        ordered = (
            (gains or losses or different or pool)
            if task == "exam"
            else (different or pool)
        )
        i = ordered[0]
        r = rows[i]
        folder = MODEL / "examples" / task
        folder.mkdir(parents=True, exist_ok=True)
        q = save_strip(folder, "question", bitmap(ink, r["q"]))
        blocks = [
            save_strip(folder, "region-" + str(j), bitmap(ink, v))
            for j, v in enumerate(r["blocks"])
        ]
        dump(folder / "input.json", dict(question=q, regions=blocks))
        fields = [
            ("Question", q, r["q"]),
            (
                "Human answer",
                save_strip(folder, "reference", bitmap(ink, r["answers"][0])),
                r["answers"][0],
            ),
        ]
        for m, l in zip(
            ARMS,
            ("Previous reply", "Independent reply", "Competing reply"),
            strict=True,
        ):
            a = runs[m][i]
            raw = bytes.fromhex(a["ink_hex"])
            assert sha256(raw).hexdigest() == a["sha256"]
            fields.append((l, save_strip(folder, m, raw), a["text"]))
        y = panel * 135
        plate.label(5, y + 2, "ab"[panel], 12, True)
        plate.label(23, y + 2, title, 10, True)
        e = [breaks(v) for _, _, v in fields]
        dy = min(18, 100 / sum(len(v) - 1 for v in e))
        y += 19
        for (label, spec, _), edges in zip(fields, e, strict=True):
            plate.label(
                6, y, label, 7.4, True, TEAL if label == "Competing reply" else MUTED
            )
            for line, (a, b) in enumerate(zip(edges, edges[1:])):
                if b > a:
                    plate.strip(
                        f"{panel}-{spec['png']}-{line}",
                        folder / spec["png"],
                        spec["cells"],
                        96,
                        y + line * dy,
                        440,
                        dy,
                        a,
                        b,
                    )
            y += (len(edges) - 1) * dy + 2
        chosen.append(
            dict(
                index=i,
                task=task,
                question=r["q"],
                answers=r["answers"],
                arms={
                    m: {
                        k: runs[m][i][k]
                        for k in ("text", "fact_text", "sha256", "fact_sha256", "stop")
                    }
                    for m in ARMS
                },
            )
        )
    plate.rule(132)
    dump(PUB / "plates.json", [plate.render()])
    dump(PUB / "examples.json", chosen)
    d = read(PUB / "scores.json")
    trace = {
        m: [r["mean_loss"] for r in read(MODEL / m / "fit.json")["epochs"]]
        for m in ARMS[1:]
    }
    dump(PUB / "trajectory.json", trace)
    x = np.arange(len(TASKS))
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(7.65, 3.0), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    for j, (mode, name, color) in enumerate(
        zip(
            ARMS,
            ("Previous", "Independent", "Competing"),
            ("#8b7970", "#597ba2", TEAL),
            strict=True,
        )
    ):
        a.bar(
            x + 0.24 * (j - 1),
            [
                100 * d[mode]["free"][t]["exact"] / d[mode]["free"][t]["n"]
                for t in TASKS
            ],
            0.23,
            color=color,
            label=name,
            zorder=3,
        )
        if mode in trace:
            b.plot(
                range(1, 9),
                trace[mode],
                label=name,
                color=color,
                marker="o" if j == 2 else "s",
                ms=3,
                ls="-" if j == 2 else "--",
            )
    a.set(
        xticks=x,
        xticklabels=("Exam", "Story", "MLQA\nEnglish", "MLQA\nChinese", "DROP"),
        ylabel="Exact stopped replies (%)",
    )
    b.set(
        xticks=[1, 2, 4, 6, 8],
        xlabel="Pass over supported training lists",
        ylabel="Mean list loss (nats)",
    )
    for ax, title in (
        (a, "a   Writing on unseen pages"),
        (b, "b   Scoring the evidence"),
    ):
        ax.set_title(title, loc="left", fontsize=10, weight="bold")
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dfe7e5", lw=0.6)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(w_pad=1.3)
    for ext in ("png", "pdf"):
        fig.savefig(
            FIG / ("Figure_180_writing_after_candidate_comparison." + ext), dpi=220
        )
    plt.close(fig)


def story_pair():
    rows = read(WORK / "confirmation.json")
    judged = read(PUB / "semantic.json")
    indices = read(PUB / "comparison.json")["indices"]
    raw = unpack(WORK / "competing.json.gz")["original"]
    j = next(
        j
        for j, i in enumerate(indices)
        if rows[i]["task"] == "stories"
        and raw[i]["stop"]
        and all(judged["votes"][2 * j + k]["competing"] for k in range(2))
    )
    first = indices[j]
    ink = Ink(np.load(BASE / "palette.npy"))
    plate = Plate("Figure_181_two_questions_in_a_story", PUB / "plates")
    selected = []
    for c, i in enumerate((first, first ^ 1)):
        r, o = rows[i], raw[i]
        x = 274 * c
        folder = MODEL / "examples" / f"story-question-{c+1}"
        folder.mkdir(parents=True, exist_ok=True)
        q = save_strip(folder, "question", bitmap(ink, r["q"]))
        blocks = [
            save_strip(folder, "region-" + str(j), bitmap(ink, b))
            for j, b in enumerate(r["blocks"])
        ]
        dump(folder / "input.json", dict(question=q, regions=blocks))
        human = save_strip(folder, "reference", bitmap(ink, r["answers"][0]))
        im = bytes.fromhex(o["ink_hex"])
        assert sha256(im).hexdigest() == o["sha256"]
        answer = save_strip(folder, "competing", im)
        plate.label(x + 4, 4, "ab"[c], 12, True)
        plate.label(
            x + 22,
            4,
            ("A credited story reply", "The other question on the same page")[c],
            10,
            True,
        )
        for k, (label, ref) in enumerate(
            (
                ("Pictured question", q),
                ("Human reference", human),
                ("Generated reply", answer),
            )
        ):
            y = 22 + 34 * k
            plate.label(x + 5, y, label, 7.5, True, TEAL if k == 2 else MUTED)
            plate.strip(
                f"{c}-{k}", folder / ref["png"], ref["cells"], x + 5, y + 10, 258, 19
            )
        selected.append(
            dict(
                index=i,
                question=r["q"],
                reference=r["answers"],
                generated=o["text"],
                sha256=o["sha256"],
                fact_sha256=o["fact_sha256"],
            )
        )
    scene = plate.render()
    dump(
        PUB / "plates.json",
        [v for v in read(PUB / "plates.json") if v["name"] != scene["name"]] + [scene],
    )
    dump(PUB / "story-pair.json", selected)


if __name__ == "__main__":
    make()
    story_pair()
