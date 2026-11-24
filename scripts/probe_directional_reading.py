"""Exchange left/right only inside the learned correction, or exchange the question."""

from collections import Counter, defaultdict
from contextlib import contextmanager
from hashlib import sha256
import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from scripts.prepare_directional_reading import WORK, PUB, read, dump
from scripts.evaluate_directional_reading import models, bitmap
from scripts.evaluate_visual_discourse import unpack
from scripts.prepare_fact_channel import MODEL as BASE, Ink
from scripts.train_region_writer import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


@contextmanager
def exchanged_context(actor, mode):
    if mode == "original":
        yield
        return
    head = actor.reader.correction

    def exchange(module, args):
        if mode == "unsigned":
            return args
        value = args[0].clone()
        d = head.query.out_features
        for width in range(len(head.radii)):
            j = (2 + 2 * width) * d
            value[..., j : j + d].neg_()
        return (value,)

    h = head.combine.register_forward_pre_hook(exchange)
    try:
        yield
    finally:
        h.remove()


def run():
    rows = read(WORK / "confirmation.json")
    ink = Ink(np.load(BASE / "palette.npy"))
    qs = [ink.encode(r["q"]) for r in rows]
    ps = [[ink.encode(s) for s in r["blocks"]] for r in rows]
    refs = [{bitmap(ink, s) for s in r["answers"]} for r in rows]
    tiles = ink.array()
    features = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
    summary = {}
    with gpu_owner(), torch.inference_mode():
        for mode, ctor in models().items():
            net = ctor()
            saved = unpack(WORK / (mode + ".json.gz"))["original"]
            groups = defaultdict(Counter)
            for i in range(0, len(rows), 12):
                q = qs[i : i + 12]
                pages = ps[i : i + 12]
                signed = net.select(features, q, pages)
                with exchanged_context(net, mode):
                    reverse = net.select(features, q, pages)
                sibling = net.select(features, [q[j ^ 1] for j in range(len(q))], pages)
                for j, (a, b, c) in enumerate(
                    zip(signed, reverse, sibling, strict=True), i
                ):
                    x, y, z = [
                        tiles[np.asarray(s[2], int)].tobytes() for s in (a, b, c)
                    ]
                    if sha256(x).hexdigest() != saved[j]["fact_sha256"]:
                        raise ValueError("Different original fact")
                    if mode != "signed" and x != y:
                        raise ValueError("Exchanged an invariant view")
                    groups[rows[j]["task"]].update(
                        n=1,
                        intact=x in refs[j],
                        exchange=y in refs[j],
                        lost_on_exchange=x in refs[j] and y not in refs[j],
                        gained_on_exchange=x not in refs[j] and y in refs[j],
                        changed_on_exchange=x != y,
                        changed_question=z != x,
                        sibling_on_original=z in refs[j],
                    )
            summary[mode] = groups
    dump(PUB / "side-intervention.json", summary)
    print(summary)


if __name__ == "__main__":
    configure_cuda_cli()
    run()
