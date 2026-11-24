"""Replay released QA PNGs and challenge the same reader with historical source ink."""

from collections import defaultdict
from hashlib import sha256

import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import DEFAULT_FONT, read_image, read_text
from scripts.evaluate_visual_discourse import draw
from scripts.prepare_paired_evidence import ROOT, MODEL, PUB, dump, read
from scripts.write_paired_evidence import generate, reader
from scripts.write_visual_discourse import Writer, encode_pages


@torch.inference_mode()
def predict(writer, net, items):
    pages = [page for pair in items for page in pair]
    shapes, seq = encode_pages(writer.palette, pages)
    features = torch.from_numpy(observations({"tiles": shapes}, "ink"))
    qs, ss = seq[::2], seq[1::2]
    outputs = generate(
        writer.model,
        net,
        features,
        len(writer.palette),
        qs,
        ss,
        [s + q for s, q in zip(ss, qs, strict=True)],
        96,
    )
    return [(shapes[np.asarray(ids, dtype=int)], end) for ids, end in outputs]


def replay(writer, net):
    rows = []
    for file in sorted((MODEL / "examples").glob("*/*/[01]/readout.json")):
        row = read(file)
        source = file.parent
        question = read_image(source / "query.png", row["query_cells"])
        passage = read_image(source / "passage.png", row["passage_cells"])
        ((ink, stop),) = predict(writer, net, [(question, passage)])
        digest = sha256(ink.tobytes()).hexdigest()
        if digest != row["sha256"] or stop != row["stopped"]:
            raise ValueError(f"Independent CPU PNG replay differs at {source}")
        rows.append(
            dict(example=str(source.relative_to(MODEL)), sha256=digest, stopped=stop)
        )
    if len(rows) != 16:
        raise ValueError("Expected sixteen released examples")
    dump(PUB / "image-replay.json", rows)


STAGES = {
    "oracle": ("甲骨文", "oracle-bone"),
    "bronze": ("金文", "bronze"),
    "seal": ("篆书", "seal"),
    "liushutong": ("六书通", "Liushutong"),
}


def historical(writer, net):
    archive = ROOT / "data/source_plates_v1"
    sheets = read(archive / "development_cards.json")
    pixels = np.load(archive / "pixels.npy", mmap_mode="r")
    groups = defaultdict(dict)
    for sheet in sheets:
        for r in sheet:
            if r["stage"] in STAGES:
                groups[r["family"]].setdefault(r["stage"], r)
    selected = [g for _, g in sorted(groups.items()) if len(g) >= 2][:64]
    yan = {
        r["stage"]: r
        for r in read(archive / "yan/source.json")["rows"]
        if r["char"] == "言"
    }
    records = []
    for i, g in enumerate([*selected, yan]):
        rows = [g[k] for k in STAGES if k in g][:2]
        if i % 2:
            rows.reverse()
        headword = rows[0]["char"]
        items = []
        metadata = []
        qtext = []
        for lang in (0, 1):
            names = [STAGES[r["stage"]][lang] for r in rows]
            text = (
                f"字书记载，{headword}的{names[0]}字形是§，{names[1]}字形是§。"
                if not lang
                else f"In this dictionary, the {names[0]} form of {headword} is §. Its {names[1]} form is §."
            )
            parts = text.split("§")
            source = np.concatenate(
                [
                    read_text(parts[0], DEFAULT_FONT),
                    pixels[[rows[0]["tile"]]],
                    read_text(parts[1], DEFAULT_FONT),
                    pixels[[rows[1]["tile"]]],
                    read_text(parts[2], DEFAULT_FONT),
                ]
            )
            for r, name in zip(rows, names, strict=True):
                question = (
                    f"文中{headword}的{name}字形是什么？"
                    if not lang
                    else f"Which mark is the {name} form of {headword}?"
                )
                items.append((read_text(question, DEFAULT_FONT), source))
                metadata.append((lang, r))
                qtext.append(question)
        for (ink, end), (lang, r), (query, page), question in zip(
            predict(writer, net, items), metadata, items, qtext, strict=True
        ):
            expected = pixels[[r["tile"]]]
            exact = end and np.array_equal(ink, expected)
            tag = "en" if lang else "zh"
            records.append(
                dict(
                    headword=headword,
                    question=question,
                    language=tag,
                    label=r["label"],
                    stage=r["stage"],
                    source_svg_sha256=r["svg"],
                    emitted=len(ink),
                    stopped=end,
                    expected_sha256=sha256(expected.tobytes()).hexdigest(),
                    output_sha256=sha256(ink.tobytes()).hexdigest(),
                    correct=exact,
                    example=i == len(selected),
                )
            )
            if i == len(selected):
                dest = MODEL / "examples/yan" / tag / r["stage"]
                dest.mkdir(parents=True, exist_ok=True)
                for key, arr in (
                    ("query", query),
                    ("passage", page),
                    ("answer", ink),
                    ("reference", expected),
                ):
                    draw(dest, key, arr)
                dump(
                    dest / "readout.json",
                    dict(records[-1], query_cells=len(query), passage_cells=len(page)),
                )
    totals = {}
    for lang in ("zh", "en"):
        pairs = defaultdict(list)
        for r in records:
            if r["language"] == lang and not r["example"]:
                pairs[r["headword"]].append(r["correct"])
        if any(len(v) != 2 for v in pairs.values()) or len(pairs) != 64:
            raise ValueError("Incomplete historical pairs")
        totals[lang] = dict(
            pairs=len(pairs),
            both=sum(all(v) for v in pairs.values()),
            individual=sum(sum(v) for v in pairs.values()),
        )
    dump(PUB / "historical.json", dict(summary=totals, rows=records))
    print(totals, "言", [r["correct"] for r in records[-4:]])


if __name__ == "__main__":
    torch.set_num_threads(3)
    writer = Writer(device="cpu")
    net = reader("evidence", "cpu")
    replay(writer, net)
    historical(writer, net)
