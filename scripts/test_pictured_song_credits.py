"""Record both requested sentence orders on the diagnostic song and a new exchanged song."""

import fcntl
import json
import os
import numpy as np
import torch
from ilm.visual_lm.bound_fact_reader import BoundFactReader
from ilm.visual_lm.fact_realizer import FactRealizer
from ilm.visual_lm.pictured_relation import PicturedRelation
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.prepare_visual_facts import Face, FONTS, ROOT, WORK, RELATIONS, write_json
from scripts.train_visual_facts import READER
from scripts.train_fact_realizer import target
from scripts.fit_pictured_relation import OUT, EVIDENCE
from scripts.train_causal_glyph_writer import WORK as GPT


def run():
    face = Face((FONTS[0], 2))
    rows = [
        (
            "秋潮",
            "新歌《秋潮》由蓝禾作词，沈河作曲。",
            "《秋潮》的作曲人是谁？",
            "作曲",
            "沈河",
        )
    ]
    for lyricist, composer in (("苏恬", "周至"), ("周至", "苏恬")):
        p = f"新曲《松窗》由{lyricist}作词，{composer}作曲。"
        rows.extend(
            [
                ("松窗", p, "谁是《松窗》的作曲人？", "作曲", composer),
                ("松窗", p, "谁为《松窗》写下歌词？", "作词", lyricist),
            ]
        )
    reader = BoundFactReader(READER, WORK / "genuine", "cuda")
    writer = FactRealizer(WORK / "writer", "cuda")
    channel = PicturedRelation(OUT / "genuine.npz")
    pairs = [(face(p), face(q)) for _, p, q, _, _ in rows]
    orig = reader.fields_many(pairs)
    new = []
    for (s, r, o), (_, q) in zip(orig, pairs):
        new.append((s, channel.choose(q, s, r)[0], o))
    legend = dict(
        zip(
            writer.writer.ink,
            json.loads((GPT / "audit_legend.json").read_text()),
            strict=True,
        )
    )
    for s, p, q, r, o in rows:
        legend.update({face(c): c for c in s + p + q + r + o})
    result = []
    for k in range(len(rows)):
        s, p, q, r, o = rows[k]
        result.append(
            dict(
                key=k,
                page=p,
                question=q,
                reference=dict(s=s, r=r, o=o),
                baseline_relation=RELATIONS[orig[k][1]],
                updated_relation=RELATIONS[new[k][1]],
            )
        )
    for arm, fields in (("baseline", orig), ("updated", new)):
        for j, order in enumerate(("subject", "object")):
            for i, (y, stop) in enumerate(writer.answer_fields(fields, order)):
                rec = result[i]
                exact = stop and y == face(target(rec["reference"], j))
                rec[arm + "_" + order] = dict(
                    text="".join(
                        legend.get(y[k : k + 128], "□") for k in range(0, len(y), 128)
                    ),
                    exact=exact,
                    stop=stop,
                )
                for tail, raw in (
                    (arm + "_" + order, y),
                    ("question", pairs[i][1]),
                    ("passage", pairs[i][0]),
                ):
                    write_strip(
                        np.frombuffer(raw, np.uint8).reshape(-1, 128),
                        OUT / f"song_{i}_{tail}.png",
                    )
    write_json(EVIDENCE / "songs.json", result)
    print(
        [
            (
                r["updated_object"]["text"],
                r["updated_subject"]["exact"],
                r["updated_object"]["exact"],
            )
            for r in result
        ]
    )


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Use assigned GPU 0")
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(3)
        run()
