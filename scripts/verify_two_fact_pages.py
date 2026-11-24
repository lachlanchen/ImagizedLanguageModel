"""Replay all original held PNGs on CPU and preserve a success and the first factual failure."""

import gzip
import hashlib
import io
import json

import numpy as np
import torch

from scripts.prepare_chinese_two_fact import (
    MODEL,
    PUB,
    WORK,
    ROOT,
    Rasterizer,
    read,
    dump,
)
from scripts.write_chinese_two_fact import TwoFactReply, load_page, card_image, display
from scripts.write_chinese_card_reply import pack


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cursor", action="store_true")
    cursor = ap.parse_args().cursor
    torch.set_num_threads(2)
    model = TwoFactReply(arm="cursor" if cursor else "ink-joined")
    raster = Rasterizer()
    held = read(WORK / "test.json")
    reference = read(PUB / "alignment-ink.json")["test"]["all_lines"]
    if cursor:
        reference = read(PUB / "cursor-results.json")["ink"]["test"]["lines"]
    direct = json.loads(gzip.decompress((PUB / "replies.json.gz").read_bytes()))[
        "test"
    ]["ink-joined"]
    art = ROOT / (
        "references/evidence/field_cursor_v1"
        if cursor
        else "references/evidence/two_fact_figure_v1"
    )
    art.mkdir(exist_ok=True)
    folder = MODEL / ("cursor-examples" if cursor else "examples")
    folder.mkdir(exist_ok=True)
    count = 0
    streams = hashlib.sha256()
    examples = {}
    different = []
    for i, (r, expected) in enumerate(zip(held, reference, strict=True)):
        lines = [r["q"], r["domain"], "名称：" + r["name"]] + [
            k + "：" + v for k, v in r["card"].items()
        ]
        page = card_image(np.stack([pack(line, raster) for line in lines]))
        with io.BytesIO() as buffer:
            page.save(buffer, format="PNG")
            buffer.seek(0)
            bits = load_page(buffer)
        plan, out = model.reply(bits)
        raw = out.tobytes()
        streams.update(len(out).to_bytes(4, "big"))
        streams.update(raw)
        same = (
            raw == b"".join(raster.tile(c) for c in expected["line"])
            and plan == expected["plan"]
        )
        count += int(same)
        if not same:
            different.append(dict(index=i, plan=plan, line=model.transcript(out)))
        if len(r["keys"]) < 2:
            continue
        name = (
            "first"
            if "first" not in examples
            else "failure"
            if not expected["clean"] and "failure" not in examples
            else None
        )
        if cursor and i == 526:
            name = "exchange"
        if name:
            examples[name] = dict(
                index=i,
                question=r["q"],
                name=r["name"],
                source=r["card"],
                reference=r["answer"],
                route=plan,
                output=model.transcript(out),
                original=direct[i]["line"],
            )
            page.save(folder / (name + "-card.png"))
            display(out).save(folder / (name + "-answer.png"))
            (folder / (name + "-answer.bin")).write_bytes(raw)
            if cursor and name == "first":
                continue
            display(bits[0, : len(r["q"])], 24).save(art / (name + "-query.png"))
            display(out, 36 if cursor else 28).save(art / (name + "-answer.png"))
            if cursor and name == "exchange":
                older = TwoFactReply(arm="ink-joined")
                _, old_ink = older.reply(bits)
                assert old_ink.tobytes() == b"".join(
                    raster.tile(c)
                    for c in read(PUB / "alignment-ink.json")["test"]["all_lines"][i][
                        "line"
                    ]
                )
                display(old_ink, 36).save(art / "exchange-before.png")
                examples[name]["earlier_aligned"] = older.transcript(old_ink)
            if name == "first":
                parent = TwoFactReply(arm="ink-joined", aligned=False)
                _, before = parent.reply(bits, cover=False)
                if before.tobytes() != b"".join(
                    raster.tile(c) for c in direct[i]["line"]
                ):
                    raise ValueError("Original output changed")
                display(before, 28).save(art / "first-original.png")
    dump(
        PUB / ("cursor-physical.json" if cursor else "physical.json"),
        dict(
            identical=count,
            n=len(held),
            stream_sha256=streams.hexdigest(),
            disagreements=different,
            examples=examples,
        ),
    )
    print(count, len(held), examples)


if __name__ == "__main__":
    main()
