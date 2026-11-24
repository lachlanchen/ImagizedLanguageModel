"""Exchange target and distractor fields while holding the glyph multiset fixed."""

from itertools import combinations
from collections import Counter
import gzip
import json
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_listener import beam_write
from scripts.prepare_chinese_card_reply import (
    MODEL,
    WORK,
    PUB,
    FIELDS,
    DOMAINS,
    WIDTH,
    Rasterizer,
    read,
    dump,
)
from scripts.train_chinese_card_reply import router, output_bank, load_writer
from scripts.train_semantic_page import gpu_owner
from scripts.write_chinese_card_reply import pack
from scripts.evaluate_chinese_card_reply import grade

FRAMES = {
    "single": "请问{target}是多少？",
    "exclude_first": "不用告诉我{other}，我只想知道{target}。",
    "exclude_last": "我想查{target}，不是{other}。",
    "discussed": "刚刚说过{other}了，请告诉我{target}。",
}


def main():
    held = read(WORK / "test.json")
    cases = []
    seen = set()
    for r in held:
        for a, b in combinations(sorted(r["card"]), 2):
            key = r["domain"], a, b
            if key in seen or r["card"][a] == r["card"][b]:
                continue
            seen.add(key)
            for frame, text in FRAMES.items():
                qa = text.format(target=a, other=b)
                qb = text.format(target=b, other=a)
                if frame != "single" and Counter(qa) != Counter(qb):
                    raise ValueError("Different glyph multisets")
                for q, target in ((qa, a), (qb, b)):
                    cases.append(dict(r, q=q, key=target, frame=frame, group=key))
    raster = Rasterizer()
    chars = read(MODEL / "readback.json")
    report = []
    summary = {}
    with gpu_owner(), torch.inference_mode():
        out = output_bank("cuda")
        for mode in ("ink", "byte"):
            net = load_writer(mode, "cuda")

            def observe(lines):
                if mode == "ink":
                    packed = np.stack([pack(line, raster) for line in lines])
                    obs = observations({"tiles": packed.reshape(-1, 128)}, "ink")
                else:
                    codes = np.zeros((len(lines), WIDTH, 4), np.uint8)
                    for i, line in enumerate(lines):
                        for j, c in enumerate(line):
                            codes[i, j, : len(c.encode())] = list(c.encode())
                    obs = observations({"bytes": codes.reshape(-1, 4)}, "byte")
                return torch.as_tensor(obs, device="cuda").reshape(
                    len(lines), WIDTH, 256
                )

            labels = net.encode_rows(observe(list(FIELDS) + list(DOMAINS)))
            for j in range(0, len(cases), 32):
                batch = cases[j : j + 32]
                q = observe([r["q"] for r in batch])
                domain = torch.tensor(
                    [list(DOMAINS).index(r["domain"]) for r in batch], device="cuda"
                )
                avail = torch.tensor(
                    [[k in r["card"] for k in FIELDS] for r in batch], device="cuda"
                )
                route, mem = router(net, q, labels, domain, avail)
                ids, ns = beam_write(
                    net,
                    mem,
                    torch.ones(mem.shape[:2], dtype=torch.bool, device="cuda"),
                    out,
                    None,
                    limit=120,
                )
                for row, seq, n, k in zip(
                    batch, ids.tolist(), ns.tolist(), route.argmax(1).tolist()
                ):
                    g = grade(row, FIELDS[k], seq[:n], chars)
                    report.append(
                        dict(
                            mode=mode,
                            group=row["group"],
                            frame=row["frame"],
                            q=row["q"],
                            requested=row["key"],
                            selected=FIELDS[k],
                            written=g["line"],
                            correct=g["clean_value"] and g["route"],
                        )
                    )
            for frame in FRAMES:
                subset = [
                    r for r in report if r["mode"] == mode and r["frame"] == frame
                ]
                summary[mode + "-" + frame] = dict(
                    pairs=len(subset) // 2,
                    both=sum(
                        a["correct"] and b["correct"]
                        for a, b in zip(subset[::2], subset[1::2])
                    ),
                    correct=sum(r["correct"] for r in subset),
                )
    dump(PUB / "focus.json", summary)
    (PUB / "focus-sentences.json.gz").write_bytes(
        gzip.compress(json.dumps(report, ensure_ascii=False).encode(), mtime=0)
    )
    print(summary)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
