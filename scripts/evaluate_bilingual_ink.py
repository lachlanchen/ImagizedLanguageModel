"""Autonomous translation, paired semantic controls, held retrieval and reverse decoding."""

from collections import Counter, defaultdict
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file
from ilm.visual_lm.bilingual_alignment import BilingualTraining, read_sentence
from ilm.visual_lm.ordered_translation import OrderedTranslator
from ilm.visual_lm.reverse_translation import rerank
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_bilingual_ink import BASE, MODEL, WORK, PUB, read, dump, sha
from scripts.prepare_proposition_translation import Ink, key
from scripts.evaluate_proposition_translation import (
    cohort as previous,
    model as original,
    semantic,
)
from scripts.train_ordered_translation import pack
from scripts.write_ordered_translation import stopped
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.measure_writer_energy import snapshot, consumed

MODES = ("parent", "likelihood", "aligned", "forward", "roundtrip")


def model(mode, device):
    if mode == "parent":
        return original("likelihood", device)
    d = MODEL / mode
    if sha(d / "weights.safetensors") != read(d / "fit.json")["weights"]:
        raise ValueError(mode)
    net = OrderedTranslator(1)
    net.load_state_dict(load_file(str(d / "weights.safetensors")))
    return net.to(device).eval()


def cohort():
    rows = previous()
    extra = read(WORK / "held.json")
    for i, r in enumerate(extra[:32]):
        for t in (0, 1):
            rows.append(
                dict(
                    group="opus",
                    kind="human",
                    id=f"opus-{i}/{t}",
                    variant=0,
                    t=t,
                    source=r["text"][1 - t],
                    ref=r["text"][t],
                    foil=extra[(i + 16) % 32]["text"][t],
                )
            )
    return rows


@torch.inference_mode()
def evaluate(rows):
    render = Ink(np.load(BASE / "palette.npy"))
    src = [render.encode(r["source"]) for r in rows]
    refs = [render.encode(r["ref"]) for r in rows]
    if render.bad:
        raise ValueError(render.bad)
    f = torch.tensor(observations({"tiles": render.array()}, "ink"), device="cuda")
    lookup = read(BASE / "readback.json")
    inv = [torch.tensor(v, device="cuda") for v in read(BASE / "inventories.json")]
    out = [{} for _ in rows]
    retrieval = {}
    cost = {}
    extra = read(WORK / "held.json")

    def record(i, m, c, stop):
        out[i][m] = dict(
            text="".join(lookup[str(j)] for j in c),
            stop=stop,
            cells=c,
            raw_exact=stop and c == refs[i],
        )

    for m in MODES[:3]:
        net = model(m, "cuda")
        wrap = BilingualTraining(net).cuda().eval()
        if m == "aligned":
            p = MODEL / m / "alignment-projection.safetensors"
            if not p.is_file():
                p = WORK / "alignment-projection.safetensors"
            if sha(p) != read(MODEL / m / "fit.json")["training_projection"]:
                raise ValueError("Projection differs")
            wrap.project.load_state_dict(load_file(str(p)))
        bank = net.optical(f)
        means = []
        for t in (0, 1):
            mat = pack([r["cells"][t] for r in extra], "cuda")
            _, _, h = read_sentence(net, bank, mat)
            means.append(h)
        variants = {"raw": means}
        if m == "aligned":
            variants["projected"] = [wrap.project(a) for a in means]
        for k, arr in variants.items():
            a, b = [torch.nn.functional.normalize(h.float(), dim=-1) for h in arr]
            sim = a @ b.T
            gold = torch.arange(len(extra), device="cuda")
            retrieval[m + "/" + k] = dict(
                n=len(extra),
                en_to_zh=int((sim.argmax(1) == gold).sum()),
                zh_to_en=int((sim.argmax(0) == gold).sum()),
            )
        modes_cost = defaultdict(float)
        for t in (0, 1):
            ids = [i for i, r in enumerate(rows) if r["t"] == t]
            for j in range(0, len(ids), 16):
                batch = ids[j : j + 16]
                xs = [src[i] for i in batch]
                meter = snapshot()
                start = monotonic()
                pred = net.generate(f, pack(xs, "cuda"), inv[t], t)
                greedy = [stopped(y) for y in pred.cpu()]
                modes_cost["greedy_seconds"] += monotonic() - start
                modes_cost["greedy_joules"] += consumed(meter, snapshot())["joules"]
                for i, (cells, stop) in zip(batch, greedy, strict=True):
                    record(i, m, cells, stop)
                if m == "aligned":
                    meter = snapshot()
                    start = monotonic()
                    alternatives = rerank(net, f, xs, inv, t, greedy)
                    modes_cost["extra_seconds"] += monotonic() - start
                    modes_cost["extra_joules"] += consumed(meter, snapshot())["joules"]
                    for i, a in zip(batch, alternatives, strict=True):
                        for method in MODES[3:]:
                            record(i, method, a[method], True)
                        out[i]["search"] = dict(
                            candidates=len(a["candidates"]),
                            exact_in_list=refs[i] in a["candidates"],
                            normalized_in_list=any(
                                key("".join(lookup[str(k)] for k in s))
                                == key(rows[i]["ref"])
                                for s in a["candidates"]
                            ),
                        )
        cost[m] = dict(modes_cost, n=len(rows))
    return out, retrieval, cost


def main():
    configure_cuda_cli()
    if sha(WORK / "held.json") != read(PUB / "corpus.json")["files"]["held.json"]:
        raise ValueError("Holdout differs")
    rows = cohort()
    with gpu_owner():
        torch.set_num_threads(2)
        out, retrieval, cost = evaluate(rows)
        dump(WORK / "generated.json", dict(outputs=out, retrieval=retrieval, cost=cost))
        torch.cuda.empty_cache()
        ids = [i for i, r in enumerate(rows) if r["group"] != "combination"]
        report = semantic([rows[i] for i in ids], [out[i] for i in ids], MODES)
        exact = defaultdict(Counter)
        beam = defaultdict(Counter)
        for i, row in enumerate(rows):
            g = exact[row["group"]]
            g["n"] += 1
            for m in MODES:
                g[m] += int(out[i][m]["raw_exact"])
            b = beam[row["group"]]
            b["n"] += 1
            for k, v in out[i]["search"].items():
                b[k] += int(v)
        dump(WORK / "graded.json", dict(rows=rows, outputs=out))
        dump(
            PUB / "result.json",
            dict(
                semantic=report,
                indices=ids,
                exact=exact,
                search=beam,
                cost=cost,
                retrieval=retrieval,
                outputs=sha(WORK / "graded.json"),
            ),
        )
        print(report["groups"], flush=True)


if __name__ == "__main__":
    main()
