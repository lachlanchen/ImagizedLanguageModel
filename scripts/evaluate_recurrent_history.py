"""Replay the registered historical-page challenge with the composing optical readers."""

import numpy as np
import torch
from scripts.prepare_recurrent_evidence import ROOT, MODEL, PUBLIC, read, dump
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.replay_paired_evidence import STAGES
from scripts.write_recurrent_evidence import load, generate
from scripts.write_visual_discourse import Writer, encode_pages
from scripts.evaluate_visual_discourse import draw
from scripts.train_span_choice import gpu_owner
from ilm.visual_lm.book_prefix import observations
from hashlib import sha256


def main():
    archive = ROOT / "data/source_plates_v1"
    old = read(ROOT / "references/evidence/paired_evidence_v1/historical.json")["rows"]
    entries = [
        r for sheet in read(archive / "development_cards.json") for r in sheet
    ] + read(archive / "yan/source.json")["rows"]
    loc = {r["raster"]: r["tile"] for r in entries}
    tiles = np.load(archive / "pixels.npy", mmap_mode="r")
    ras = Rasterizer()

    def ink(t):
        return np.stack([np.frombuffer(ras.tile(c), dtype=np.uint8) for c in t])

    pages = []
    targets = []
    for i in range(0, len(old), 2):
        a, b = old[i : i + 2]
        en = a["language"] == "en"
        names = [STAGES[r["stage"]][int(en)] for r in (a, b)]
        z = a["headword"]
        s = (
            f"In this dictionary, the {names[0]} form of {z} is §. Its {names[1]} form is §."
            if en
            else f"字书记载，{z}的{names[0]}字形是§，{names[1]}字形是§。"
        ).split("§")
        imgs = [tiles[[loc[r["expected_sha256"]]]] for r in (a, b)]
        page = np.concatenate([ink(s[0]), imgs[0], ink(s[1]), imgs[1], ink(s[2])])
        for r, img in zip((a, b), imgs, strict=True):
            assert sha256(img.tobytes()).hexdigest() == r["expected_sha256"]
            pages.extend([ink(r["question"]), page])
            targets.append(img)
    with gpu_owner():
        writer = Writer(device="cuda")
        shapes, seq = encode_pages(writer.palette, pages)
        x = torch.from_numpy(observations({"tiles": shapes}, "ink")).cuda()
        for arm in ("fixed", "dynamic"):
            net = load(arm, "cuda")
            out = []
            for i in range(0, len(old), 20):
                q = seq[2 * i : 2 * (i + 20) : 2]
                s = seq[2 * i + 1 : 2 * (i + 20) : 2]
                out.extend(
                    generate(
                        writer.model,
                        net,
                        x,
                        len(writer.palette),
                        q,
                        s,
                        [a + b for a, b in zip(s, q)],
                        96,
                    )
                )
            results = []
            for i, (r, (ids, end)) in enumerate(zip(old, out, strict=True)):
                physical = shapes[np.asarray(ids, dtype=int)]
                results.append(
                    dict(
                        label=r["label"],
                        headword=r["headword"],
                        lang=r["language"],
                        example=r["example"],
                        correct=end and np.array_equal(physical, targets[i]),
                        stopped=end,
                        output=sha256(physical.tobytes()).hexdigest(),
                    )
                )
                if r["example"]:
                    dest = MODEL / "examples" / arm / "yan" / r["language"] / r["stage"]
                    dest.mkdir(parents=True, exist_ok=True)
                    for key, arr in (
                        ("question", pages[i * 2]),
                        ("passage", pages[i * 2 + 1]),
                        ("reference", targets[i]),
                        ("answer", physical),
                    ):
                        draw(dest, key, arr)
                    dump(
                        dest / "readout.json",
                        dict(
                            results[-1],
                            written=len(ids),
                            question_cells=len(pages[i * 2]),
                            passage_cells=len(pages[i * 2 + 1]),
                        ),
                    )
            dump(PUBLIC / (arm + "-historical.json"), results)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
