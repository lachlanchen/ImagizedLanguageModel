"""Ask the fitted writer to choose its own 96-cell window on an unseen full story."""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import argparse
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.evaluate_evidence_views import groups, official_index
from scripts.prepare_evidence_views import Alphabet, MODEL, PUB
from scripts.prepare_open_exam import read, dump
from scripts.train_visual_cursor import pad
from scripts.train_paired_evidence import gpu_owner
from scripts.write_open_exam import load
from scripts.write_recurrent_evidence import generate
from scripts.write_visual_discourse import Writer


def probe(root, report):
    judge, _, _, dev, _ = official_index()
    source = {
        (d["id"], a["turn_id"]): a for d in read(dev)["data"] for a in d["answers"]
    }
    data = groups(judge, dev)
    a = Alphabet()
    rows = []
    for group in ("coqa", "composition"):
        for r in data[group]:
            answer = source[r["doc"], r["turn"]]
            lo, hi = answer["span_start"], answer["span_end"]
            if (
                0 <= lo < hi <= len(r["page"])
                and r["page"][lo:hi] == answer["span_text"]
            ):
                rows.append(
                    dict(
                        r,
                        group=group,
                        q=a.encode(r["question"]),
                        p=a.encode(r["page"]),
                        lo=lo,
                        hi=hi,
                    )
                )
    x = torch.from_numpy(observations({"tiles": a.array()}, "ink")).cuda()
    outcomes = defaultdict(list)
    with gpu_owner(), torch.inference_mode():
        writer = Writer(device="cuda")
        net = load("highlight", "cuda", root)
        bank = net.ink(x)
        for i in range(0, len(rows), 12):
            rs = rows[i : i + 12]
            qs = [r["q"] for r in rs]
            ps = [r["p"] for r in rs]
            q, p = [torch.as_tensor(pad(z), device="cuda").long() for z in (qs, ps)]
            # The evidence logits are the writer's learned distribution over visible source positions.
            _, ev, _, _ = net.read(bank, q, p)
            prob = ev.double().softmax(-1).cpu().numpy()
            starts = defaultdict(list)
            for j, r in enumerate(rs):
                n = len(r["p"])
                width = min(96, n)
                hi = r["hi"]
                lo = r["lo"]
                cs = np.concatenate(([0.0], np.cumsum(prob[j, :n])))
                starts["learned"].append(int(np.argmax(cs[width:] - cs[:-width])))
                digest = sha256(
                    (r["doc"] + str(r["turn"]) + "window").encode()
                ).digest()
                starts["random"].append(
                    int.from_bytes(digest[:8], "big") % (n - width + 1)
                )
                starts["human_centered"].append(
                    min(n - width, max(0, (lo + hi - width) // 2))
                )
            for mode, offsets in starts.items():
                windows = [r["p"][k : k + 96] for r, k in zip(rs, offsets, strict=True)]
                outputs = generate(
                    writer.model,
                    net,
                    x,
                    a.n,
                    qs,
                    windows,
                    [w + q for w, q in zip(windows, qs)],
                    64,
                    geometry=(320, 4096),
                )
                for r, k, (ids, end) in zip(rs, offsets, outputs, strict=True):
                    lo, hi = r["lo"], r["hi"]
                    overlap = max(0, min(hi, k + 96) - max(lo, k)) / (hi - lo)
                    text = "".join(a.chars[j] for j in ids)
                    ok = end and judge.normalize_answer(text) in list(
                        map(judge.normalize_answer, r["refs"])
                    )
                    outcomes[r["group"], mode].append(
                        dict(doc=r["doc"], hit=ok, overlap=overlap)
                    )
    summary = defaultdict(dict)
    for (g, k), rs in outcomes.items():
        bydoc = defaultdict(list)
        for r in rs:
            bydoc[r["doc"]].append(r["hit"])
        pairs = [p for p in bydoc.values() if len(p) == 2]
        summary[g][k] = dict(
            turns=len(rs),
            exact=sum(r["hit"] for r in rs),
            pairs=len(pairs),
            both=sum(all(p) for p in pairs),
            complete_span=sum(r["overlap"] == 1 for r in rs),
            mean_span_recall=float(np.mean([r["overlap"] for r in rs])),
        )
    dump(report / "window.json", dict(budget=96, scores=summary))


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--report", type=Path, default=PUB)
    a = p.parse_args()
    probe(a.model_dir, a.report)
