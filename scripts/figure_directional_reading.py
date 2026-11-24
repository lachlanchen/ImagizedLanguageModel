"""Display original held output pixels and the measured directional intervention."""

from collections import Counter
from hashlib import sha256

import matplotlib.pyplot as plt
import numpy as np

from scripts.evaluate_directional_reading import bitmap
from scripts.evaluate_visual_discourse import draw, unpack
from scripts.figure_phrase_region import breaks
from scripts.figure_recurrent_evidence import FIG, MUTED, TEAL, Plate
from scripts.prepare_directional_reading import MODEL, PUB, WORK, dump, read
from scripts.prepare_fact_channel import MODEL as BASE, Ink

MODES = ("original", "unsigned", "signed")
TASKS = ("mlqa-en", "mlqa-zh", "drop")


def contains(outer, inner):
    return bool(inner) and any(
        outer[i : i + len(inner)] == inner
        for i in range(0, len(outer) - len(inner) + 1, 128)
    )


def extent(raw, references):
    if raw in references:
        return "exact"
    if any(contains(raw, reference) for reference in references):
        return "contains_reference"
    if any(contains(reference, raw) for reference in references):
        return "part_of_reference"
    return "other"


def save_strip(folder, key, raw):
    cells = np.frombuffer(raw, np.uint8).reshape(-1, 128)
    draw(folder, key, cells)
    return dict(png=key + ".png", cells=len(cells))


def originals():
    rows = read(WORK / "confirmation.json")
    runs = {m: unpack(WORK / (m + ".json.gz"))["original"] for m in MODES}
    ink = Ink(np.load(BASE / "palette.npy"))
    refs = [{bitmap(ink, v) for v in row["answers"]} for row in rows]
    counts = {}
    photos = {}
    scores = read(PUB / "scores.json")
    for mode, records in runs.items():
        counts[mode] = {}
        photos[mode] = []
        for ref, row, out in zip(refs, rows, records, strict=True):
            raw = bytes.fromhex(out["fact_hex"])
            if sha256(raw).hexdigest() != out["fact_sha256"]:
                raise ValueError("Evidence digest mismatch")
            category = extent(raw, ref)
            photos[mode].append(category)
            counts[mode].setdefault(row["task"], Counter()).update([category])
        for t, c in counts[mode].items():
            assert (
                sum(c.values()) == 192
                and c["exact"] == scores[mode]["confirmation"][t]["fact"]
            )
    dump(PUB / "extent.json", counts)
    assets = PUB / "plates"
    assets.mkdir(exist_ok=True)
    plate = Plate("Figure_176_reading_the_evidence_in_order", assets)
    selected = []
    for panel, (task, gain, title) in enumerate(
        (
            ("mlqa-en", True, "An occupation in an English passage"),
            ("mlqa-zh", False, "A consequence in a Chinese passage"),
        )
    ):
        i = next(
            i
            for i, r in enumerate(rows)
            if r["task"] == task
            and (photos["signed"][i] == "exact") == gain
            and (photos["original"][i] == "exact") != gain
            and max(
                len(r["q"]),
                len(r["answers"][0]),
                runs["original"][i]["cells"],
                runs["signed"][i]["cells"],
            )
            <= 120
        )
        r = rows[i]
        folder = MODEL / "examples" / task
        folder.mkdir(exist_ok=True, parents=True)
        q = save_strip(folder, "question", bitmap(ink, r["q"]))
        regions = [
            save_strip(folder, f"region-{n}", bitmap(ink, s))
            for n, s in enumerate(r["blocks"])
        ]
        dump(folder / "input.json", dict(question=q, regions=regions))
        ref = r["answers"][0]
        fields = [
            ("Question", q, r["q"]),
            ("Human answer", save_strip(folder, "reference", bitmap(ink, ref)), ref),
        ]
        for mode in MODES:
            out = runs[mode][i]
            for field, key, label in (
                ("fact", "fact", "Evidence"),
                ("ink", "answer", "Reply"),
            ):
                raw = bytes.fromhex(out[field + "_hex"])
                digest = out["fact_sha256" if field == "fact" else "sha256"]
                if sha256(raw).hexdigest() != digest:
                    raise ValueError(digest)
                spec = save_strip(folder, mode + "-" + key, raw)
                if mode != "unsigned":
                    fields.append(
                        (
                            label
                            + (" · earlier" if mode == "original" else " · directed"),
                            spec,
                            out["fact_text" if field == "fact" else "text"],
                        )
                    )
        start = panel * 135
        plate.label(4, start + 2, "ab"[panel], 12, True)
        plate.label(22, start + 2, title, 10, True)
        edges = [breaks(s) for _, _, s in fields]
        dy = min(16, 100 / sum(len(e) - 1 for e in edges))
        y = start + 18
        for (label, spec, _), e in zip(fields, edges, strict=True):
            plate.label(
                5, y, label, 7.4, True, TEAL if label.endswith("directed") else MUTED
            )
            for n, (l, h) in enumerate(zip(e, e[1:])):
                plate.strip(
                    f"{panel}-{spec['png']}-{n}",
                    folder / spec["png"],
                    spec["cells"],
                    97,
                    y + dy * n,
                    442,
                    dy,
                    l,
                    h,
                )
            y += dy * (len(e) - 1) + 2
        selected.append(
            dict(
                index=i,
                task=task,
                source_id=r["source_id"],
                reference=ref,
                question=r["q"],
                arms={
                    m: {
                        k: runs[m][i][k]
                        for k in ("fact_text", "fact_sha256", "text", "sha256", "stop")
                    }
                    for m in MODES
                },
            )
        )
    plate.rule(132)
    dump(PUB / "plates.json", [plate.render()])
    dump(PUB / "examples.json", selected)


def credited_replies():
    """Show the first semantic success in each language in the fixed review order."""
    rows = read(WORK / "confirmation.json")
    raw = unpack(WORK / "signed.json.gz")["original"]
    cohort = read(PUB / "comparison.json")["indices"]
    votes = read(PUB / "semantic.json")["votes"]
    ink = Ink(np.load(BASE / "palette.npy"))
    plate = Plate("Figure_178_two_complete_visual_replies", PUB / "plates")
    chosen = []
    for col, (task, heading) in enumerate(
        (("mlqa-en", "An author"), ("mlqa-zh", "A population share"))
    ):
        i = next(
            i
            for j, i in enumerate(cohort)
            if rows[i]["task"] == task
            and raw[i]["stop"]
            and votes[2 * j]["signed"]
            and votes[2 * j + 1]["signed"]
        )
        r, o = rows[i], raw[i]
        folder = MODEL / "examples" / (task + "-credited")
        folder.mkdir(parents=True, exist_ok=True)
        q = save_strip(folder, "question", bitmap(ink, r["q"]))
        blocks = [
            save_strip(folder, f"region-{k}", bitmap(ink, v))
            for k, v in enumerate(r["blocks"])
        ]
        dump(folder / "input.json", dict(question=q, regions=blocks))
        ref = save_strip(folder, "human", bitmap(ink, r["answers"][0]))
        im = bytes.fromhex(o["ink_hex"])
        assert sha256(im).hexdigest() == o["sha256"]
        out = save_strip(folder, "answer", im)
        x = 275 * col
        plate.label(x + 4, 4, "ab"[col], 12, True)
        plate.label(x + 22, 4, heading, 10, True)
        for j, (name, spec) in enumerate(
            (("Pictured question", q), ("Human reference", ref), ("Model reply", out))
        ):
            y = 21 + 32 * j
            plate.label(x + 6, y, name, 7.4, True, TEAL if j == 2 else MUTED)
            plate.strip(
                f"{col}-{j}",
                folder / spec["png"],
                spec["cells"],
                x + 6,
                y + 10,
                258,
                17,
            )
        chosen.append(
            dict(
                index=i,
                task=task,
                reference=r["answers"],
                question=r["q"],
                model=o["text"],
                sha256=o["sha256"],
                fact_sha256=o["fact_sha256"],
            )
        )
    scene = plate.render()
    plates = [r for r in read(PUB / "plates.json") if r["name"] != scene["name"]]
    dump(PUB / "plates.json", plates + [scene])
    dump(PUB / "credited.json", chosen)


def trends():
    scores = read(PUB / "scores.json")
    trace = {
        m: [r["mean_nll"] for r in read(MODEL / m / "fit.json")["epochs"]]
        for m in MODES[1:]
    }
    dump(PUB / "trajectory.json", trace)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.05))
    x = np.arange(3)
    for j, (mode, label, color) in enumerate(
        zip(
            MODES,
            ("Earlier", "Unsigned", "Signed"),
            ("#877a70", "#587da1", TEAL),
            strict=True,
        )
    ):
        values = [scores[mode]["confirmation"][t]["fact"] for t in TASKS]
        bars = a.bar(
            x + (j - 1) * 0.25, values, 0.235, color=color, label=label, zorder=3
        )
        a.bar_label(bars, fontsize=7, padding=2)
        if mode in trace:
            b.plot(
                [1, 2, 3, 4],
                trace[mode],
                marker=("s" if j == 1 else "o"),
                ls=("--" if j == 1 else "-"),
                color=color,
                label=label,
                lw=1.7,
                ms=4,
            )
    a.set(
        xticks=x,
        xticklabels=("English", "Chinese", "DROP"),
        ylim=(0, 54),
        ylabel="Exact evidence / 192 questions",
    )
    b.set(
        xticks=(1, 2, 3, 4),
        xlabel="Complete pass over 39,057 questions",
        ylabel="Mean training loss (nats)",
    )
    for ax, title in (
        (a, "a   Evidence on new passages"),
        (b, "b   Learning to locate evidence"),
    ):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e3e7e6", lw=0.7)
        ax.tick_params(labelsize=8)
        ax.legend(frameon=False, fontsize=7, ncol=3 if ax == a else 1)
    fig.tight_layout(pad=1.1, w_pad=2)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("Figure_177_learning_to_locate_evidence." + ext), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    originals()
    credited_replies()
    trends()
