"""Independently replay PNG pages on CPU and preserve byte-exact emitted ink."""

import base64
import gzip
import hashlib
import io
import json
import zipfile
from collections import Counter
import numpy as np
import torch
from scripts.prepare_prose_fields import BASE, PUBLIC, WEIGHTS, read, dump, norm
from scripts.train_prose_fields import train_rows
from scripts.prose_entity_data import pairs, page
from scripts.write_chinese_card_reply import card_image, display
from scripts.write_prose_fields import ProseReply, render, load_png
from scripts.prepare_chinese_card_reply import Rasterizer


def png_io(page):
    with io.BytesIO() as buf:
        card_image(page).save(buf, format="PNG")
        buf.seek(0)
        return load_png(buf)


def empty_or_match(x, y):
    if (
        x["keys"] != y["keys"]
        or x["missing"] != y["missing"]
        or {k: list(v) for k, v in x["intervals"].items()} != y["intervals"]
    ):
        return False
    return (
        None
        if x["pixels"] is None
        else hashlib.sha256(x["pixels"].tobytes()).hexdigest()
    ) == y["ink"]


def main():
    torch.set_num_threads(3)
    d = json.loads(gzip.decompress((PUBLIC / "entity-decisions.json.gz").read_bytes()))
    old = json.loads(gzip.decompress((PUBLIC / "decisions.json.gz").read_bytes()))
    r = Rasterizer()
    runs = {}
    physical = {}
    indices = {}
    samples = WEIGHTS / "examples"
    samples.mkdir(exist_ok=True)
    model = ProseReply(arm="entity-ink")
    training, n = train_rows()
    fit_text = {norm(x["text"]) for x in training}
    train_raw = read(BASE / "train-passages.json")
    for i, (a, b, k) in enumerate(pairs(train_raw, 8701)):
        left, h = (a, b) if i % 2 else (b, a)
        fit_text.add(norm(page(left, h, left, k)["text"]))
    original = ProseReply()
    with zipfile.ZipFile(PUBLIC / "test-images.zip") as z:
        for group in (
            "test-natural",
            "test-assembled",
            "test-missing",
            "authored",
            "edits",
        ):
            rows = old["rows"][group]
            saved = old["results"]["ink"][group]["actual"]
            count = 0
            for i in range(0, len(rows), 32):
                ps = [
                    load_png(io.BytesIO(z.read(f"{group}/{j:04}-page.png")))
                    for j in range(i, min(i + 32, len(rows)))
                ]
                outputs = original.batch(ps)
                for j, (out, ref) in enumerate(
                    zip(outputs, saved[i : i + 32], strict=True)
                ):
                    if not empty_or_match(out, ref):
                        raise ValueError(("Original public PNG mismatch", group, i + j))
                    if out["pixels"] is not None:
                        from PIL import Image

                        with Image.open(
                            io.BytesIO(z.read(f"{group}/{i+j:04}-answer.png"))
                        ) as img:
                            if not np.array_equal(
                                np.asarray(img), np.asarray(display(out["pixels"]))
                            ):
                                raise ValueError("Wrong archived PNG")
                    count += 1
            runs["base/" + group] = count
    for g in ("test", "native-multi", "authored-multi"):
        rows = d["rows"][g]
        refs = d["results"]["entity-ink"][g]
        stream = []
        counts = Counter()
        for i in range(0, len(rows), 32):
            rs = rows[i : i + 32]
            if any(norm(x["text"]) in fit_text for x in rs):
                raise ValueError("Training paragraph collision")
            ps = [
                png_io(render(x["q"], x["domain"], x["name"], x["text"], r)) for x in rs
            ]
            output = model.batch(ps)
            for j, (rec, p, x, y) in enumerate(
                zip(rs, ps, output, refs[i : i + 32], strict=True)
            ):
                if not empty_or_match(x, y):
                    raise ValueError(("Entity PNG mismatch", g, i + j))
                pix = x["pixels"]
                h = y["ink"]
                if pix is not None:
                    physical[h] = base64.b64encode(pix.tobytes()).decode()
                stream.append(h)
                for key, (left, b) in rec["spans"].items():
                    if rec["text"][left:b] != rec["card"][key]:
                        raise ValueError("Reference offset mismatch")
                if g == "native-multi" or (g == "authored-multi" and i + j in (4, 5)):
                    name = (
                        ("restaurant-" + str(i + j + 1))
                        if g == "native-multi"
                        else ("park-" + str(i + j - 3))
                    )
                    card_image(p).save(samples / (name + "-page.png"))
                    if pix is not None:
                        display(pix).save(samples / (name + "-answer.png"))
                        (samples / (name + "-answer.bin")).write_bytes(pix.tobytes())
                counts["cases"] += 1
        indices[g] = stream
        runs["entity/" + g] = counts["cases"]
    packed = json.dumps(
        dict(
            encoding="base64 of 128 packed bytes per glyph, canonical answer strip",
            streams=indices,
            bitmaps=physical,
        ),
        separators=(",", ":"),
    ).encode()
    (PUBLIC / "entity-test-ink.json.gz").write_bytes(gzip.compress(packed, mtime=0))
    dump(
        PUBLIC / "replay.json",
        dict(
            identical=runs,
            distinct_output_strips=len(physical),
            competing_test_passages=len(set(row["text"] for row in d["rows"]["test"])),
            train_passage_collisions=0,
        ),
    )
    print(runs, flush=True)


if __name__ == "__main__":
    main()
