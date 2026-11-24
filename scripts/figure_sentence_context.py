"""Publish physical answer strips, free-writing trajectories and released examples."""

from hashlib import sha256
import numpy as np
import matplotlib.pyplot as plt
from scripts.prepare_sentence_context import WORK, PUB, MODEL, Ink, read, dump
from scripts.evaluate_visual_discourse import unpack, draw
from scripts.figure_recurrent_evidence import Plate, FIG, TEAL, MUTED
from scripts.figure_phrase_region import breaks


def pictures():
    modes = ("original", "direct", "surround")
    rows = read(WORK / "confirmation.json")
    answers = {m: unpack(WORK / (m + ".json.gz"))["original"] for m in modes}
    ink = Ink(np.load(MODEL / "palette.npy"))

    def pixels(t):
        return b"".join(ink.tiles[k] for k in ink.encode(t))

    def valid(o, r):
        return o["stop"] and any(pixels(s).hex() == o["ink_hex"] for s in r["answers"])

    assets = PUB / "plates"
    assets.mkdir(exist_ok=True)
    plate = Plate("Figure_173_writing_around_the_fact", assets)
    chosen = []
    for ix, (task, heading) in enumerate(
        (
            ("statements", "A complete English statement"),
            ("exam", "A Chinese reading question"),
        )
    ):
        pool = [
            i
            for i, r in enumerate(rows)
            if r["task"] == task
            and max(
                [len(r["q"]), len(r["answers"][0])]
                + [answers[m][i]["cells"] for m in modes]
            )
            <= 120
        ]
        # The English example prioritizes a new correct reply; the Chinese example prioritizes a lost one.
        desired = not ix
        picks = [
            i
            for i in pool
            if valid(answers["surround"][i], rows[i]) == desired
            and valid(answers["direct"][i], rows[i]) != desired
        ]
        rule = "first matched-control gain" if desired else "first matched-control loss"
        if not picks:
            picks = [
                i
                for i in pool
                if valid(answers["surround"][i], rows[i]) == desired
                and valid(answers["original"][i], rows[i]) != desired
            ]
            rule = "first old-baseline gain" if desired else "first old-baseline loss"
        if not picks:
            picks = pool
            rule = "first in size range"
        i = picks[0]
        r = rows[i]
        folder = MODEL / "examples" / task
        folder.mkdir(parents=True, exist_ok=True)

        def put(k, raw):
            im = np.frombuffer(raw, np.uint8).reshape(-1, 128)
            draw(folder, k, im)
            return dict(png=k + ".png", cells=len(im))

        dump(
            folder / "input.json",
            dict(
                question=put("question", pixels(r["q"])),
                regions=[
                    put("source-" + str(b), pixels(v))
                    for b, v in enumerate(r["blocks"])
                ],
            ),
        )
        ref = next(
            (
                s
                for s in r["answers"]
                if pixels(s).hex() == answers["surround"][i]["ink_hex"]
            ),
            r["answers"][0],
        )
        put("reference", pixels(ref))
        fact = answers["surround"][i]
        put("fact", bytes.fromhex(fact["fact_hex"]))
        fields = [
            ("Question", "question", r["q"]),
            ("Evidence read", "fact", fact["fact_text"]),
            ("Human reply", "reference", ref),
        ]
        for m, label in zip(
            modes, ("Earlier writer", "Direct fit", "Context fit"), strict=True
        ):
            a = answers[m][i]
            raw = bytes.fromhex(a["ink_hex"])
            assert sha256(raw).hexdigest() == a["sha256"]
            put(m, raw)
            fields.append((label, m, a["text"]))
        top = 135 * ix
        plate.label(4, top + 2, "ab"[ix], 12, True)
        plate.label(22, top + 2, heading, 10, True)
        lines = [breaks(txt) for _, _, txt in fields]
        dy = min(15.0, 96 / sum(len(e) - 1 for e in lines))
        y = top + 19
        for (label, k, txt), edges in zip(fields, lines, strict=True):
            plate.label(6, y, label, 7.5, True, TEAL if k == "surround" else MUTED)
            for line, (a, b) in enumerate(zip(edges, edges[1:])):
                if b > a:
                    plate.strip(
                        f"{ix}-{k}-{line}",
                        folder / (k + ".png"),
                        len(txt),
                        89,
                        y + line * dy,
                        443,
                        dy,
                        a,
                        b,
                    )
            y += dy * (len(edges) - 1) + 2
        chosen.append(
            dict(
                task=task,
                index=i,
                rule=rule,
                question=r["q"],
                evidence=fact["fact_text"],
                reference=ref,
                outputs={
                    m: {k: answers[m][i][k] for k in ("text", "sha256", "stop")}
                    for m in modes
                },
            )
        )
    plate.rule(132)
    dump(PUB / "plates.json", [plate.render()])
    dump(PUB / "examples.json", chosen)


def results():
    sc = read(PUB / "scores.json")
    tr = read(PUB / "trajectory.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.65, 3.15))
    colors = ("#88786a", "#557da1", TEAL)
    labels = ("Earlier", "Direct", "Context")
    x = np.arange(2)
    ax = axes[0]
    for i, m in enumerate(("original", "direct", "surround")):
        v = [sc[m][t]["bitmap_exact"] for t in ("statements", "exam")]
        bars = ax.bar(
            x + (i - 1) * 0.22, v, 0.21, color=colors[i], label=labels[i], zorder=2
        )
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xticks(x, ["English", "Chinese"])
    ax.set_ylabel("Exact replies / 192", fontsize=9)
    ax.set_ylim(0, 65)
    ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper left")
    ax.set_title("a   New passages", loc="left", fontsize=11, weight="bold")
    ax = axes[1]
    steps = sorted(map(int, tr))
    for t, lab, color in (
        ("statements", "English", "#557da1"),
        ("exam", "Chinese", TEAL),
    ):
        ax.plot(
            steps,
            [tr[str(j)][t]["bitmap_exact"] for j in steps],
            label=lab,
            color=color,
            marker="o",
            linewidth=1.8,
            markersize=4,
        )
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 5000, 9798], ["0", "5,000", "9,798"])
    ax.set_xlabel("Context-writer update", fontsize=9)
    ax.set_ylabel("Exact replies / 48", fontsize=9)
    ax.set_title(
        "b   Earlier development passages", loc="left", fontsize=11, weight="bold"
    )
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color="#e1e6e5", linewidth=0.6, zorder=0)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.19, top=0.87, wspace=0.38)
    for ext in ("pdf", "png"):
        fig.savefig(
            FIG / ("Figure_174_complete_answers_during_training." + ext), dpi=220
        )
    plt.close(fig)


def paired_questions():
    from scripts.reserve_visual_lexical_controls import DEST, PUBLIC
    from scripts.prepare_fact_channel import MODEL as retained

    rows = read(DEST / "confirmation.json")
    raw = unpack(DEST / "images.json.gz")["free"]
    measured = read(PUBLIC / "test.json")
    if sha256((DEST / "images.json.gz").read_bytes()).hexdigest() != measured["physical_outputs"]:
        raise ValueError("Changed search outputs")
    ix = next(i for i, r in enumerate(measured["pages"])
              if r["group"] == "exam/overlap-resistant" and r["free"] == 1
              and all(raw[2*i+j]["cells"] <= 60 for j in (0, 1)))
    ink = Ink(np.load(retained / "palette.npy"))
    plate = Plate("Figure_175_two_questions_about_one_company", PUB / "plates")
    examples = []
    for j, heading in enumerate(("An acquisition", "The origin of the company name")):
        index = 2*ix+j
        row, reply = rows[index], raw[index]
        folder = retained.parent / "fact_span_ranking/examples" / ("lexical-control-" + str(j+1))
        folder.mkdir(exist_ok=True, parents=True)
        def picture(name, cells):
            draw(folder, name, cells)
            return dict(png=name+".png", cells=len(cells))
        def render_text(name, value):
            ids = ink.encode(value)
            return picture(name, np.asarray([np.frombuffer(ink.tiles[k], np.uint8) for k in ids]))
        q = render_text("question", row["q"])
        regions = [render_text("region-"+str(k), v) for k,v in enumerate(row["blocks"])]
        dump(folder/"input.json",dict(question=q,regions=regions))
        human = render_text("reference",row["answers"][0])
        fields=[("Question",q),("Human reply",human)]
        for name,label,key in (("fact","Evidence read","fact"),("answer","Model reply","ink")):
            b = bytes.fromhex(reply[key+"_hex"])
            if sha256(b).hexdigest() != reply["fact_sha256" if key=="fact" else "sha256"]:
                raise ValueError("Changed row")
            fields.append((label,picture(name,np.frombuffer(b,np.uint8).reshape(-1,128))))
        y = 135*j
        plate.label(4,y+2,"ab"[j],12,True)
        plate.label(22,y+2,heading,10,True)
        for k,(label,ref) in enumerate(fields):
            top = y+22+24*k
            plate.label(6,top,label,8,True,TEAL if k==3 else MUTED)
            plate.strip(f"q{j}-{k}",folder/ref["png"],ref["cells"],90,top,445,20)
        examples.append(dict(index=index,question=row["q"],human=row["answers"],
                             generated=reply["text"],sha256=reply["sha256"],stopped=reply["stop"]))
    plate.rule(132)
    scene=plate.render()
    other=[v for v in read(PUB/"plates.json") if v["name"] != scene["name"]]
    dump(PUB/"plates.json",other+[scene])
    dump(PUBLIC/"examples.json",examples)


if __name__ == "__main__":
    pictures()
    results()
    paired_questions()
