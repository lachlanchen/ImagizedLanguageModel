"""Fit a pixel-based paragraph selector on real writer and human QA records."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from ilm.visual_lm.paragraph_evidence import ParagraphRanker, PhysicalBank, pad
from scripts.paragraph_evidence_corpus import WORK, corpus
from scripts.prepare_visual_facts import Face, FONTS
from scripts.span_reservation import normalized
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner


def records(folder, name):
    path = folder / (name + ".jsonl")
    info = json.loads((folder / "corpus.json").read_text())[name]
    if sha(path) != info["sha256"]:
        raise ValueError("Paragraph split differs from its preparation")
    return list(map(json.loads, path.read_text().splitlines()))


def train(folder):
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    ds, held = corpus()
    # Remove exact paragraph reuse, including cross-study/source reuse.
    seen = {normalized(p) for x in held for p in x["parts"]}
    ds = [x for x in ds if all(normalized(p) not in seen for p in x["parts"])]
    info = {}
    for name, rows in (("train", ds), ("held", held)):
        f = folder / (name + ".jsonl")
        f.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        info[name] = dict(
            sha256=sha(f), groups=dict(Counter(x["type"] for x in rows)), rows=len(rows)
        )
    record(folder / "corpus.json", info)
    print(info, flush=True)
    face = Face((FONTS[0], 2))
    bank = PhysicalBank()
    texts = {}

    def as_image(s):
        if s not in texts:
            texts[s] = bank.index(face(s))
        return texts[s]

    items = [
        (as_image(x["q"]), [as_image(p) for p in x["parts"]], x["right"]) for x in ds
    ]
    ink = bank.pooled_ink("cuda")
    seed = 241132
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = ParagraphRanker().cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    start = monotonic()
    steps = 0
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(5):
        total = 0.0
        for pool in np.array_split(
            rng.permutation(len(items)), max(1, len(items) // 384)
        ):
            order = sorted(pool, key=lambda i: max(map(len, items[i][1])))
            for ids in np.array_split(order, int(np.ceil(len(order) / 32))):
                questions, parts, targets = zip(*(items[i] for i in ids))
                qq = pad([q for q in questions for _ in range(2)], "cuda")
                pp = pad([p for pair in parts for p in pair], "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = net(ink, qq, pp).reshape(-1, 2).float()
                    loss = F.cross_entropy(logits, torch.tensor(targets, device="cuda"))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                total += float(loss.detach()) * len(ids)
                steps += 1
                if (
                    monotonic() - start > 1200
                    or torch.cuda.max_memory_allocated() > 20 * 2**30
                ):
                    raise RuntimeError(
                        "Paragraph selector exceeded its registered budget"
                    )
        losses.append(total / len(items))
        print(epoch + 1, losses[-1], flush=True)
    path = folder / "paragraph.safetensors"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()},
        str(path),
    )
    record(
        folder / "fit.json",
        dict(
            loss=losses,
            steps=steps,
            rows=len(items),
            seconds=monotonic() - start,
            seed=seed,
            parameters=sum(p.numel() for p in net.parameters()),
            sha256=sha(path),
            peak_bytes=torch.cuda.max_memory_allocated(),
            gpu=torch.cuda.get_device_name(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=WORK)
    with gpu_owner():
        train(parser.parse_args().run_dir)
