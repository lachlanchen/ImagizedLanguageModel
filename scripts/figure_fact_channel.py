"""Print original queries, intermediate facts and released autonomous answer pixels."""

from hashlib import sha256
import numpy as np
from scripts.prepare_fact_channel import MODEL, WORK, PUB, Ink, read, dump
from scripts.evaluate_visual_discourse import unpack, draw
from scripts.evaluate_region_writer import changed
from scripts.figure_recurrent_evidence import Plate, TEAL, MUTED, FIG
from scripts.figure_phrase_region import breaks


def examples():
    rows = read(WORK / "confirmation.json")
    edits = changed(rows)
    d = {a: unpack(WORK / ("fresh-" + a + ".json.gz")) for a in ("direct", "predicted")}
    ink = Ink(np.load(MODEL / "palette.npy"))

    def pixels(s):
        return b"".join(ink.tiles[k] for k in ink.encode(s))

    def correct(a, side, i):
        v = d[a][side][i]
        r = (rows if side == "original" else edits)[i]
        return v["stop"] and bytes.fromhex(v["ink_hex"]) in [
            pixels(s) for s in r["answers"]
        ]

    assets = PUB / "plates"
    assets.mkdir(exist_ok=True)
    records = []
    plates = []
    for t, file, title in (
        ("statements", "Figure_168_reading_before_writing", "An English statement"),
        ("exam", "Figure_169_following_a_pictured_fact", "A Chinese reading answer"),
    ):
        candidates = [i for i, r in enumerate(rows) if r["task"] == t]

        def choice(a, b, excluded):
            available = [i for i in candidates if i not in excluded]
            return next(
                (
                    i
                    for i in available
                    if correct(a, "original", i) and not correct(b, "original", i)
                ),
                next(
                    (i for i in available if not correct("predicted", "original", i)),
                    available[0],
                ),
            )

        first = choice("predicted", "direct", [])
        selected = [
            ("original", first),
            ("original", choice("direct", "predicted", [first])),
        ]
        if t == "exam":
            pair = next(
                (
                    i
                    for i, r in enumerate(edits)
                    if r["task"] == t
                    and correct("predicted", "edited", i)
                    and correct("predicted", "original", r["original"])
                ),
                None,
            )
            if pair is not None:
                selected = [("original", edits[pair]["original"]), ("edited", pair)]
        plate = Plate(file, assets)
        for j, (side, i) in enumerate(selected):
            r = (rows if side == "original" else edits)[i]
            p = d["predicted"][side][i]
            base = d["direct"][side][i]
            ref = next(
                (
                    s
                    for s in r["answers"]
                    if pixels(s).hex() in (p["ink_hex"], base["ink_hex"])
                ),
                r["answers"][0],
            )
            folder = MODEL / "examples" / (t + "-" + str(j + 1))
            folder.mkdir(parents=True, exist_ok=True)

            def raster(name, b):
                arr = np.frombuffer(b, np.uint8).reshape(-1, 128)
                draw(folder, name, arr)
                return dict(cells=len(arr), png=name + ".png")

            query = raster("question", pixels(r["q"]))
            bs = [
                raster("source-" + str(k), pixels(s)) for k, s in enumerate(r["blocks"])
            ]
            dump(folder / "input.json", dict(question=query, regions=bs))
            raster("reference", pixels(ref))
            raster("fact", bytes.fromhex(p["fact_hex"]))
            for a, v in (("direct", base), ("predicted", p)):
                raw = bytes.fromhex(v["ink_hex"])
                assert sha256(raw).hexdigest() == v["sha256"]
                raster(a, raw)
            exact = correct("predicted", side, i)
            heading = (
                title + " after a fact substitution"
                if side == "edited"
                else title + " from the original page"
            )
            top = 135 * j
            plate.label(3, top + 1, "ab"[j], 12, True)
            plate.label(22, top + 1, heading, 10, True)
            fields = [
                ("Question", "question", r["q"]),
                ("Fact read", "fact", p["fact_text"]),
                ("Human reply", "reference", ref),
                ("Full page", "direct", base["text"]),
                ("Fact + region", "predicted", p["text"]),
            ]
            spans = [
                breaks(s)
                if t == "statements"
                else [*range(0, max(len(s), 1), 40), len(s)]
                for _, _, s in fields
            ]
            dy = min(17, (103 - 3 * 4) / sum(len(z) - 1 for z in spans))
            y = top + 18
            for (label, key, s), edges in zip(fields, spans):
                plate.label(
                    6,
                    y + 0.5,
                    label,
                    7.8,
                    True,
                    TEAL if key in ("fact", "predicted") else MUTED,
                )
                if not s:
                    plate.label(84, y, "[empty]", 8)
                else:
                    for l, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                        plate.strip(
                            f"{j}-{key}-{l}",
                            folder / (key + ".png"),
                            len(s),
                            84,
                            y + dy * l,
                            450,
                            dy,
                            lo,
                            hi,
                        )
                y += dy * (len(edges) - 1) + 3
            records.append(
                dict(
                    task=t,
                    folder=folder.name,
                    side=side,
                    index=i,
                    question=r["q"],
                    human=ref,
                    fact=p["fact_text"],
                    region=p["selected"],
                    fact_hash=p["fact_sha256"],
                    candidate=p["text"],
                    direct=base["text"],
                    candidate_hash=p["sha256"],
                    direct_hash=base["sha256"],
                    stopped=p["stop"],
                    candidate_exact=exact,
                    direct_exact=correct("direct", side, i),
                )
            )
        plate.rule(132)
        plates.append(plate.render())
    dump(PUB / "plates.json", plates)
    dump(PUB / "examples.json", records)


def trajectory():
    import matplotlib.pyplot as plt

    rc = read(PUB / "trajectory.json")
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 4.35), sharex=True)
    for col, (t, title) in enumerate(
        (("statements", "English statements"), ("exam", "Chinese reading questions"))
    ):
        for row, (metric, name) in enumerate(
            (("bitmap_exact", "Exact answer image"), ("fact", "Requested fact present"))
        ):
            ax = axes[row, col]
            for arm, label, color, marker in (
                ("direct", "Full page", "#885640", "s"),
                ("human", "Fact alone", "#446796", "^"),
                ("predicted", "Fact + region", "#006b66", "o"),
            ):
                steps = sorted(map(int, rc[arm]))
                counts = [rc[arm][str(s)]["totals"][t][metric] for s in steps]
                ax.plot(
                    steps,
                    counts,
                    label=label,
                    color=color,
                    marker=marker,
                    ms=4,
                    lw=1.35,
                    alpha=0.9,
                )
            if not row:
                ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
            if not col:
                ax.set_ylabel(name + " (/64)", fontsize=9)
            if row:
                ax.set_xlabel("Optimizer updates", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#e2e7e5", lw=0.6)
            ax.tick_params(labelsize=8)
            ax.set_xticks([0, 5000, 10000, 12260], ["0", "5,000", "10,000", "12,260"])
            ax.set_ylim(bottom=-1)
    for a, b in axes:
        top = max(a.get_ylim()[1], b.get_ylim()[1])
        a.set_ylim(-1, top)
        b.set_ylim(-1, top)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        ncol=3,
        loc="lower center",
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.09, right=0.98, top=0.91, bottom=0.19, wspace=0.23, hspace=0.26
    )
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("Figure_170_learning_to_state_a_fact." + ext), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    examples()
    trajectory()
