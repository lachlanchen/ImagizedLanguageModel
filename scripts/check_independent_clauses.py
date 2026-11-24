"""Write each inferred field independently using the same final optical question plan."""

import torch
import numpy as np
from scripts.prepare_chinese_two_fact import (
    PUB,
    WORK,
    PARENT,
    FIELDS,
    DOMAINS,
    WIDTH,
    Rasterizer,
    read,
    dump,
)
from scripts.evaluate_chinese_two_fact import independent, grade, totals, surfaces
from scripts.train_chinese_card_reply import load_writer, output_bank
from scripts.train_semantic_page import gpu_owner
from ilm.visual_lm.book_prefix import observations
from scripts.train_field_alignment import load_alignment


@torch.inference_mode()
def main():
    records = [r for r in read(WORK / "test.json") if len(r["keys"]) == 2]
    with gpu_owner():
        parent = load_writer("ink", "cuda")
        reader = load_alignment("ink", "cuda")
        palette = output_bank("cuda")
        labels = np.load(PARENT / "label-pixels.npy")
        states = parent.encode_rows(
            torch.as_tensor(
                observations({"tiles": labels.reshape(-1, 128)}, "ink"), device="cuda"
            ).reshape(len(labels), WIDTH, 256)
        )
        query = torch.as_tensor(surfaces(records, "ink", Rasterizer()), device="cuda")
        dom = torch.tensor(
            [list(DOMAINS).index(r["domain"]) for r in records], device="cuda"
        )
        available = torch.tensor(
            [[k in r["card"] for k in FIELDS] for r in records], device="cuda"
        )
        plans = reader.choose(query, dom, available).tolist()
        results = independent(
            parent,
            states,
            query,
            dom,
            available,
            plans,
            palette,
            records,
            read(PARENT / "readback.json"),
        )
        lines = [
            grade(s, [FIELDS[i] for i in plan if i < len(FIELDS)], r)
            for s, plan, r in zip(results, plans, records)
        ]
        if [r["plan"] for r in lines] != [
            r["plan"] for r in read(PUB / "alignment-ink.json")["test"]["lines"]
        ]:
            raise ValueError("Comparator plans differ")
    dump(PUB / "aligned-separate.json", dict(lines=lines, summary=totals(lines)))


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
