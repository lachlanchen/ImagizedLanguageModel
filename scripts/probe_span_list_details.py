"""Test if changing a year can reach the writer and the independently conditioned reader."""

import json
import re
import torch

from ilm.visual_lm.fact_realizer import FactRealizer
from ilm.visual_lm.span_list import SpanListReader
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.evaluate_span_list import contains
from scripts.prepare_screened_span_list import CACHE, PUBLIC
from scripts.prepare_visual_facts import Face, FONTS, RELATIONS, WORK, write_json
from scripts.span_reservation import reservation
from scripts.train_pointing_causal_writer import sha
from scripts.train_span_choice import gpu_owner, model
from scripts.train_span_list import WEIGHTS
from scripts.train_visual_facts import rows


@torch.inference_mode()
def run():
    if (PUBLIC / "details.json").exists():
        raise FileExistsError("Date comparison already frozen")
    face = Face((FONTS[0], 2))
    hp = WEIGHTS / "wide.safetensors"
    assert (
        sha(hp)
        == json.loads((PUBLIC / "fit.json").read_text())["arms"]["wide"]["weights"]
    )
    reader = SpanListReader(model(), hp)
    writer = FactRealizer(WORK / "writer", "cuda")
    q = face("这段文字中唯一出现的四位年份是什么？")
    records = []
    for v in rows("held", eligible_only=True):
        if v["group"] != "duie":
            continue
        years = re.findall(r"(?<![0-9])([12][0-9]{3})年", v["p"])
        if (
            len(years) != 1
            or len(re.findall(r"(?<![0-9])[0-9]{4}(?![0-9])", v["p"])) != 1
        ):
            continue
        old = years[0]
        alt = str(int(old) + 1)
        if old in v["s"] or old in v["o"] or alt in v["p"]:
            continue
        edited = v["p"].replace(old + "年", alt + "年", 1)
        if reservation().overlaps(edited):
            continue
        records.append((v, edited, old, alt))
        if len(records) == 32:
            break
    pages = []
    fields = []
    years = []
    for v, edited, old, alt in records:
        triple = (face(v["s"]), RELATIONS.index(v["r"]), face(v["o"]))
        for p, good, bad in ((v["p"], old, alt), (edited, alt, old)):
            pages.append((face(p), q))
            fields.append(triple)
            years.append((face(good), face(bad)))
    supplied = writer.answer_fields(fields, "natural")
    direct = [(o, True) for _, _, o in reader.fields_many(pages)]
    stats = {}
    for arm, outs in [("fields_writer", supplied), ("page_reader", direct)]:
        ok = [
            bool(stop and contains(y, good) and not contains(y, bad))
            for (y, stop), (good, bad) in zip(outs, years, strict=True)
        ]
        stats[arm] = dict(
            n=len(outs),
            correct=sum(ok),
            both=sum(a and b for a, b in zip(ok[::2], ok[1::2], strict=True)),
            equal_outputs=sum(
                a[0] == b[0] for a, b in zip(outs[::2], outs[1::2], strict=True)
            ),
            stopped=sum(stop for _, stop in outs) if arm == "fields_writer" else None,
        )
        store_streams(
            CACHE / (arm + "_dates.npz"),
            [y for y, _ in outs],
            [stop for _, stop in outs],
        )
    assert stats["fields_writer"]["both"] == 0
    write_json(
        PUBLIC / "details.json",
        dict(
            pairs=len(records),
            arms=stats,
            held_keys=[v["key"] for v, _, _, _ in records],
            reader_weights=sha(hp),
        ),
    )
    print(stats, flush=True)


if __name__ == "__main__":
    with gpu_owner():
        run()
