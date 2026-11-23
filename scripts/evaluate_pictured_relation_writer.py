"""Check the released image path after correcting the relation from the question."""

from collections import Counter, defaultdict
import fcntl
import os
import torch
from ilm.visual_lm.bound_fact_reader import BoundFactReader, render_fact
from ilm.visual_lm.fact_realizer import FactRealizer
from ilm.visual_lm.pictured_relation import PicturedRelation
from scripts.prepare_visual_facts import Face, FONTS, ROOT, WORK, write_json
from scripts.train_visual_facts import rows, READER
from scripts.train_fact_realizer import target
from scripts.fit_pictured_relation import OUT, EVIDENCE
from scripts.evaluate_grounded_propositions import store_streams


def evaluate():
    face = Face((FONTS[0], 2))
    held = rows("held")
    reader = BoundFactReader(READER, WORK / "genuine", "cuda")
    writer = FactRealizer(WORK / "writer", "cuda")
    channel = PicturedRelation(OUT / "genuine.npz")
    counters = defaultdict(Counter)
    streams = [[], []]
    stops = [[], []]
    for start in range(0, len(held), 32):
        chunk = held[start : start + 32]
        pairs = [(face(x["p"]), face(x["q"])) for x in chunk]
        fields = [
            (s, channel.choose(q, s, r)[0], o)
            for (s, r, o), (_, q) in zip(reader.fields_many(pairs), pairs, strict=True)
        ]
        written = [
            writer.answer_fields(fields, order) for order in ("subject", "object")
        ]
        for i, x in enumerate(chunk):
            good = []
            c = counters[x["group"]]
            c["n"] += 1
            s, r, o = fields[i]
            c["induced"] += render_fact(reader.frames[r], s, o) == face(target(x, 0))
            for k in range(2):
                y, st = written[k][i]
                streams[k].append(y)
                stops[k].append(st)
                ok = st and y == face(target(x, k))
                good.append(ok)
                c[("subject_first", "object_first")[k]] += ok
                c["stops"] += st
            c["both"] += all(good)
    for k, mode in enumerate(("subject", "object")):
        store_streams(OUT / (mode + ".npz"), streams[k], stops[k])
    write_json(EVIDENCE / "writing.json", counters)
    print(counters, flush=True)


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Use assigned GPU 0")
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(3)
        evaluate()
