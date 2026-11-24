"""Measure autonomous development writing across actual snapshots and changed print."""

from collections import Counter
from pathlib import Path
import torch
from PIL import ImageFont
from fontTools.ttLib import TTFont
from safetensors.torch import load_file
from scripts.prepare_sentence_context import (
    PUB,
    WORK,
    FOURTH,
    MODEL,
    Ink,
    read,
    dump,
    sha,
)
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_fact_channel import load, gpu_owner
from scripts.write_sentence_context import SentenceAnswer
from scripts.evaluate_sentence_context import models
from scripts.evaluate_fact_channel import observed
from scripts.evaluate_phrase_region import measurements
from scripts.evaluate_region_writer import e2e_scorer
from scripts.audit_region_writer import wording
from scripts.verse_gpu_runtime import configure_cuda_cli


def main():
    groups = ("statements", "exam")
    rows = read(FOURTH / "confirmation.json")
    rows = [r for t in groups for r in [p for p in rows if p["task"] == t][:48]]
    trajectory = {}
    with gpu_owner(), torch.inference_mode():
        net = SentenceAnswer("surround", "cuda")
        scorer = e2e_scorer()
        for step in (0, 2500, 5000, 7500, 9798):
            if step == 0:
                net.writer.load_state_dict(load("human", "cpu").state_dict())
            else:
                path = (
                    MODEL / "surround/weights.safetensors"
                    if step == 9798
                    else WORK / f"surround-step-{step}.safetensors"
                )
                if not path.exists():
                    continue
                net.writer.load_state_dict(load_file(str(path)))
            ink = Ink(net.palette)
            outputs = observed(net, rows, ink)
            m = measurements(rows, outputs, ink, scorer)
            trajectory[step] = {
                t: dict(
                    n=48,
                    bitmap_exact=sum(
                        v["bitmap_exact"]
                        for r, v in zip(rows, m, strict=True)
                        if r["task"] == t
                    ),
                    fact=sum(
                        v["fact"]
                        for r, v in zip(rows, m, strict=True)
                        if r["task"] == t
                    ),
                )
                for t in groups
            }
        dump(PUB / "trajectory.json", trajectory)
        del net
        new = read(WORK / "confirmation.json")
        fonts = {}
        for mode, make in models().items():
            net = make()
            res = {}
            for t in groups:
                subset = [r for r in new if r["task"] == t][:32]
                desc = (
                    (Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"), 0)
                    if t == "statements"
                    else (
                        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
                        2,
                    )
                )
                for style in ("training", "serif"):
                    ink = Ink(net.palette)
                    if style == "serif":
                        face = Rasterizer()
                        face.font = ImageFont.truetype(str(desc[0]), 26, index=desc[1])
                        with TTFont(desc[0], fontNumber=desc[1]) as ft:
                            face.coverage = set(ft.getBestCmap())
                        ink.face = face
                    outs = observed(net, subset, ink)
                    total = Counter(n=len(subset))
                    for r, a in zip(subset, outs, strict=True):
                        total["stopped"] += a["stop"]
                        total["transcribed_exact"] += bool(
                            a["stop"]
                            and wording(a["text"]) in {wording(s) for s in r["answers"]}
                        )
                        total["same_font_exact"] += bool(
                            a["stop"]
                            and any(
                                b"".join(ink.tiles[i] for i in ink.encode(s)).hex()
                                == a["ink_hex"]
                                for s in r["answers"]
                            )
                        )
                        total["missing_input_font"] += bool(
                            ink.bad.intersection(r["q"] + "".join(r["blocks"]))
                        )
                    res[t + ":" + style] = dict(total)
                res[t + ":font_sha256"] = sha(desc[0])
            fonts[mode] = res
        dump(PUB / "fonts.json", fonts)
    print(trajectory, fonts, flush=True)


if __name__ == "__main__":
    configure_cuda_cli()
    main()
