"""Evaluate a fitted focus selector with the frozen human-answer writer."""

from collections import defaultdict
from itertools import combinations
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
from scripts.evaluate_chinese_card_reply import grade, score, paired
from scripts.probe_chinese_focus import FRAMES
from scripts.learn_chinese_focus import load_focus, reroute, FRESH


def main():
    held = read(WORK / "test.json")
    raster = Rasterizer()
    cases = [dict(r, probe="human", group=i) for i, r in enumerate(held)]
    seen = set()
    for r in held:
        for a, b in combinations(sorted(r["card"]), 2):
            pair = (r["domain"], a, b)
            if pair in seen or r["card"][a] == r["card"][b]:
                continue
            seen.add(pair)
            for name, frame in FRAMES.items():
                for target, other in ((a, b), (b, a)):
                    cases.append(
                        dict(
                            r,
                            q=frame.format(target=target, other=other),
                            key=target,
                            probe=name,
                            group=pair,
                        )
                    )
            for j, (frame, k) in enumerate(FRESH):
                for x, y in ((a, b), (b, a)):
                    cases.append(
                        dict(
                            r,
                            q=frame.replace("A", x).replace("B", y),
                            key=(x, y)[k],
                            probe=f"fresh-{j+1:02}",
                            group=pair,
                        )
                    )
    states = {}
    saved = {}
    chars = read(MODEL / "readback.json")
    with gpu_owner(), torch.inference_mode():
        net = load_writer("ink", "cuda")
        focus = load_focus("cuda")
        out = output_bank("cuda")
        labels = np.load(MODEL / "label-pixels.npy")
        ink = observations({"tiles": labels.reshape(-1, 128)}, "ink")
        keys = net.encode_rows(
            torch.as_tensor(ink, device="cuda").reshape(len(labels), WIDTH, 256)
        )
        for mode in ("parent", "focused"):
            results = []
            applied = 0
            for j in range(0, len(cases), 32):
                part = cases[j : j + 32]
                pix = np.array([pack(r["q"], raster) for r in part])
                q = torch.as_tensor(
                    observations({"tiles": pix.reshape(-1, 128)}, "ink"), device="cuda"
                ).reshape(len(part), WIDTH, 256)
                dom = torch.tensor(
                    [list(DOMAINS).index(r["domain"]) for r in part], device="cuda"
                )
                av = np.array([[k in r["card"] for k in FIELDS] for r in part])
                route, mem = router(
                    net, q, keys, dom, torch.as_tensor(av, device="cuda")
                )
                pick = route.argmax(1)
                touched = 0
                if mode == "focused":
                    pick, touched = reroute(focus, pix, av, pick)
                    mem[:, 2] = keys[pick]
                applied += touched
                ids, ns = beam_write(
                    net,
                    mem,
                    torch.ones(mem.shape[:2], device="cuda", dtype=torch.bool),
                    out,
                    None,
                    limit=120,
                )
                for r, s, n, k in zip(part, ids.tolist(), ns.tolist(), pick.tolist()):
                    d = grade(r, FIELDS[k], s[:n], chars)
                    results.append(
                        d
                        | dict(
                            probe=r["probe"], group=r["group"], q=r["q"], key=r["key"]
                        )
                    )
            groups = defaultdict(list)
            for result in results:
                groups[result["probe"]].append(result)
            metrics = {}
            for name, sub in groups.items():
                if name == "human":
                    groupscore = score(sub, list(range(len(sub))))
                    pairs = paired(held)
                    groupscore["paired"] = sum(
                        sub[i]["clean_value"] and sub[j]["clean_value"]
                        for i, j in pairs
                    )
                else:
                    groupscore = dict(
                        n=len(sub) // 2,
                        both=sum(
                            left["route"]
                            and right["route"]
                            and left["clean_value"]
                            and right["clean_value"]
                            for left, right in zip(sub[::2], sub[1::2])
                        ),
                    )
                metrics[name] = groupscore
            states[mode] = metrics
            saved[mode] = results
    old = read(PUB / "focus.json")
    for name in FRAMES:
        if old["ink-" + name]["both"] != states["parent"][name]["both"]:
            raise ValueError("Original probe changed")
    dump(PUB / "focus-transfer.json", states)
    (PUB / "focus-transfer-lines.json.gz").write_bytes(
        gzip.compress(json.dumps(saved, ensure_ascii=False).encode(), mtime=0)
    )
    print(
        [
            (
                k,
                v["human"]["clean_value"],
                sum(g["both"] for n, g in v.items() if n.startswith("fresh")),
            )
            for k, v in states.items()
        ]
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
