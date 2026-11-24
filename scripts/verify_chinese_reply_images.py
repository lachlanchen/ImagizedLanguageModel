"""Replay every held question through the public PNG reader on CPU and compare actual answer ink."""

import gzip
import hashlib
import io
import json
from collections import Counter

import numpy as np
import torch

from scripts.learn_chinese_focus import mask_mentions
from scripts.prepare_chinese_card_reply import FIELDS, PUB, WORK, Rasterizer, dump, read
from scripts.write_chinese_card_reply import (
    ChineseReply,
    card_image,
    load_page,
    pack,
    transcript,
)


def main():
    rows = read(WORK / "test.json")
    with gzip.open(PUB / "outputs.json.gz") as stream:
        recorded = json.load(stream)["outputs"]
    assert len(rows) == len(recorded["ink"]) == len(recorded["byte"])
    torch.set_num_threads(2)
    engine = ChineseReply("cpu", focus=True)
    raster = Rasterizer()
    counts = Counter()
    disagreements = []
    digest = hashlib.sha256()
    for i, (row, simulated, byte) in enumerate(
        zip(rows, recorded["ink"], recorded["byte"])
    ):
        lines = [row["q"], row["domain"], "名称：" + row["name"]]
        lines.extend(key + "：" + value for key, value in row["card"].items())
        image = card_image(np.stack([pack(line, raster) for line in lines]))
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            buffer.seek(0)
            page = load_page(buffer)
        field, result = engine.reply(page)
        source = b"".join(raster.tile(c) for c in simulated["line"])
        physical = result.tobytes()
        same = source == physical and field == simulated["selection"]
        counts["pages"] += 1
        counts["identical_to_batched_gpu"] += same
        counts["exact_plain_transcript"] += transcript(result) == simulated["line"]
        counts["natural_focus_activations"] += (
            mask_mentions(page[0], [k in row["card"] for k in FIELDS]) is not None
        )
        counts["optical_only_correct"] += (
            simulated["clean_value"] and not byte["clean_value"]
        )
        counts["byte_only_correct"] += (
            byte["clean_value"] and not simulated["clean_value"]
        )
        digest.update(len(result).to_bytes(4, "big"))
        digest.update(physical)
        if not same:
            disagreements.append(
                {"index": i, "field": field, "line": transcript(result)}
            )
    dump(
        PUB / "physical-release.json",
        {
            "counts": dict(counts),
            "packed_streams_sha256": digest.hexdigest(),
            "cpu_gpu_disagreements": disagreements,
        },
    )
    print(dict(counts))


if __name__ == "__main__":
    main()
