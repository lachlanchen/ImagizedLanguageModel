"""Measure serial warm inference on identical venue questions on one device."""

from time import monotonic
import torch

from scripts.compare_two_fact_qwen import TAG, OPTIONS, ANSWER, content, local_teacher
from scripts.prepare_chinese_two_fact import WORK, PUB, Rasterizer, read, dump
from scripts.write_chinese_card_reply import pack
from scripts.write_chinese_two_fact import TwoFactReply
from scripts.evaluate_chinese_two_fact import grade
from scripts.train_semantic_page import gpu_owner, snapshot, consumed
import numpy as np


def prepared(r, raster):
    rows = [
        r["q"],
        r["domain"],
        "名称：" + r["name"],
        *(k + "：" + v for k, v in r["card"].items()),
    ]
    return np.stack([pack(s, raster) for s in rows])


def timed(items, respond):
    energy = snapshot()
    start = monotonic()
    outputs = [respond(item) for item in items]
    torch.cuda.synchronize()
    return outputs, dict(
        seconds=monotonic() - start, energy=consumed(energy, snapshot())
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cursor", action="store_true")
    parser.add_argument("--long", action="store_true")
    args = parser.parse_args()
    cursor = args.cursor
    if args.long and not cursor:
        raise ValueError("Long repeat measures the compact writer")
    out = PUB / (
        "warm-cursor-repeat.json"
        if args.long
        else "warm-cursor.json"
        if cursor
        else "warm-inference.json"
    )
    if out.exists():
        raise FileExistsError(out)
    rows = [r for r in read(WORK / "test.json") if len(r["keys"]) == 2]
    warm = next(r for r in read(WORK / "val.json") if len(r["keys"]) == 2)
    raster = Rasterizer()
    pages = [prepared(r, raster) for r in rows]
    messages = [content(r) for r in rows]
    runs = {}
    with gpu_owner():
        net = TwoFactReply("cuda", "cursor" if cursor else "ink-joined")

        def image(page, model=net):
            _, result = model.reply(page)
            return model.transcript(result)

        image(prepared(warm, raster))
        torch.cuda.synchronize()
        if args.long:
            runs["ink"] = []
            for _ in range(3):
                lines, stats = timed(pages * 32, image)
                if lines != lines[: len(rows)] * 32:
                    raise ValueError("Repeated image replies differ")
                runs["ink"].append((lines[: len(rows)], dict(stats, sweeps=32)))
        else:
            runs["ink"] = [timed(pages, image) for _ in range(2)]
        del image, net
        torch.cuda.empty_cache()
        directory = WORK / "qwen-warm"
        directory.mkdir(exist_ok=True)
        if cursor:
            earlier = read(PUB / "warm-inference.json")
            manifest = earlier["qwen_manifest"]
            runs["qwen"] = [
                (item["lines"], dict(seconds=item["seconds"], energy=item["energy"]))
                for item in earlier["passes"]["qwen"]
            ]
        else:
            with local_teacher(TAG, directory) as (ask, manifest, _):

                def text(message):
                    response = ask(
                        ANSWER,
                        dict(role="user", content=message),
                        think=False,
                        options=OPTIONS,
                    )
                    if response["stop"] != "stop":
                        raise ValueError("Incomplete response")
                    return response["text"]

                text(content(warm))
                runs["qwen"] = [timed(messages, text) for _ in range(2)]
    summary = {}
    for arm, values in runs.items():
        summary[arm] = [
            stats
            | dict(
                clean=sum(grade(s, [], r)["clean"] for s, r in zip(lines, rows)),
                lines=lines,
            )
            for lines, stats in values
        ]
    dump(out, dict(cases=len(rows), qwen_manifest=manifest, passes=summary))
    print(
        {
            arm: [(x["clean"], x["seconds"], x["energy"]["joules"]) for x in v]
            for arm, v in summary.items()
        }
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
