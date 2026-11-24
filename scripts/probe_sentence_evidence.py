"""Hold the fact, writer and question fixed while adding or removing surrounding ink."""

from collections import Counter
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_sentence_context import WORK, PUB, MODEL, read, dump, Ink
from scripts.evaluate_sentence_context import models
from scripts.evaluate_fact_channel import observed
from scripts.evaluate_visual_discourse import unpack
from scripts.train_sentence_context import extended_page
from scripts.train_region_writer import gpu_owner, pack
from scripts.verse_gpu_runtime import configure_cuda_cli


class EvidenceView:
    def __init__(self, net, full):
        self.net = net
        self.full = full
        self.chars = read(MODEL / "readback.json")

    def answer(self, tiles, qs, pages, supplied=None):
        n = self.net
        feat = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
        spans = n.select(feat, qs, pages) if supplied is None else supplied
        p = [
            extended_page(rs, f) if self.full else f
            for rs, (_, _, f) in zip(pages, spans, strict=True)
        ]
        ctx, h = n.writer.encode(feat, n.n, pack(qs, "cuda"), pack(p, "cuda"))
        return n.writer.write(ctx, h, 1024), spans


def main():
    rows = read(WORK / "confirmation.json")
    out = {}
    palette = np.load(MODEL / "palette.npy")
    with gpu_owner(), torch.inference_mode():
        for m, mk in models().items():
            net = mk()
            ink = Ink(palette)
            saved = unpack(WORK / (m + ".json.gz"))
            groups = {}
            for gold in (False, True):
                prior = saved["gold" if gold else "original"]
                full = m == "original"
                outputs = observed(EvidenceView(net, full), rows, ink, gold=gold)
                records = {}
                for name, obs in zip(
                    ("full", "strip"),
                    (outputs, prior) if full else (prior, outputs),
                    strict=True,
                ):
                    records[name] = [
                        bool(
                            a["stop"]
                            and a["ink_hex"]
                            in {
                                b"".join(ink.tiles[k] for k in ink.encode(s)).hex()
                                for s in r["answers"]
                            }
                        )
                        for a, r in zip(obs, rows, strict=True)
                    ]
                for a, b in zip(outputs, prior, strict=True):
                    if a["fact_sha256"] != b["fact_sha256"]:
                        raise ValueError("Evidence changed")
                for task in ("statements", "exam"):
                    val = Counter()
                    for i, row in enumerate(rows):
                        if row["task"] != task:
                            continue
                        f, k = [records[v][i] for v in ("full", "strip")]
                        val.update(
                            n=1,
                            full=f,
                            strip=k,
                            strip_only=k and not f,
                            full_only=f and not k,
                        )
                    groups[task + ("-gold" if gold else "-predicted")] = dict(val)
            out[m] = groups
    dump(PUB / "evidence-views.json", out)
    print(out)


if __name__ == "__main__":
    configure_cuda_cli()
    main()
