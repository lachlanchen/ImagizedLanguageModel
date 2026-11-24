"""Fit and evaluate a subword Transformer on the registered visual dialogue split."""

import argparse
import gzip
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from ilm.visual_lm.dialogue_subword import DialogueSubword
from scripts.dialogue_ink_data import read_data
from scripts.dialogue_subword_data import Turns, fit_tokenizer
from scripts.evaluate_dialogue_ink import exchanges
from scripts.evaluate_optical_passage_qa import metrics
from scripts.measure_writer_energy import snapshot, consumed
from scripts.prepare_dialogue_ink import WORK, PUB, dump, sha
from scripts.train_span_choice import gpu_owner


def timed(start, meter):
    torch.cuda.synchronize()
    return dict(
        seconds=monotonic() - start,
        energy=consumed(meter, snapshot()),
        peak_bytes=torch.cuda.max_memory_allocated(),
    )


def fit(data, folder):
    start = monotonic()
    meter = snapshot()
    tok = fit_tokenizer(data)
    ds = Turns(data, "train", tok)
    prep = timed(start, meter)
    net = DialogueSubword(tok.get_vocab_size()).cuda()
    optim = torch.optim.AdamW(net.parameters(), lr=0.00075)
    rng = np.random.default_rng(710491)
    samples = set()
    trace = []
    cost = []
    start = monotonic()
    meter = snapshot()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, 8001):
        ids = rng.integers(len(ds.rows), size=32)
        samples.update(ids)
        x, seg, label, ev = ds.batch(ids, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mem, mask, score = net.encode(x, seg)
            logits, _ = net.decode(label[:, :-1], mem, mask)
            ce = F.cross_entropy(
                logits.flatten(0, 1), label[:, 1:].flatten(), ignore_index=0
            )
            yes = ev[:, 0] >= 0
            aux = ce.new_zeros(())
            if yes.any():
                score = score.masked_fill((seg != 2)[:, :, None], -1e4)
                aux = sum(F.cross_entropy(score[yes, :, t], ev[yes, t]) for t in (0, 1))
            loss = ce + 0.2 * aux
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        optim.step()
        if step % 250 == 0:
            p = timed(start, meter)
            trace.append([step, float(ce.detach()), float(aux.detach())])
            print(step, trace[-1], p["seconds"], flush=True)
            if p["seconds"] > 3000 or p["peak_bytes"] > 12 * 2**30:
                raise RuntimeError("Exceeded the registered training ceiling")
            if step in (2000, 4000, 8000):
                cost.append(p)
    folder.mkdir(parents=True, exist_ok=False)
    tok.save(str(folder / "tokenizer.json"))
    file = folder / "transformer.safetensors"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()},
        str(file),
    )
    meta = dict(
        architecture="dialogue_subword_v1",
        parameters=sum(p.numel() for p in net.parameters()),
        vocabulary=tok.get_vocab_size(),
        model_sha256=sha(file),
        tokenizer_sha256=sha(folder / "tokenizer.json"),
        split_sha256=sha(PUB / "split.json"),
        seed=710491,
        updates=8000,
        presentations=256000,
        distinct_sampled=len(samples),
        preparation=prep,
        fit=cost[-1],
        cumulative_cost=cost,
        trace=trace,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
    )
    dump(folder / "weights.json", meta)
    return net, tok


@torch.inference_mode()
def evaluate(data, net, tok, out):
    records = []
    old = json.loads(
        gzip.decompress((WORK / "evaluation/decisions.json.gz").read_bytes())
    )
    for g in ("reserved", "dev"):
        ds = Turns(data, g, tok)
        q, p = exchanges(ds.rows)
        group = [r for r in old if r["group"] == g]
        for arm, qids, pids in [
            ("writer", None, None),
            ("exchanged_question", q, None),
            ("exchanged_passage", None, p),
        ]:
            for off in range(0, len(group), 48):
                ids = np.arange(off, min(off + 48, len(group)))
                x, seg, _, _ = ds.batch(
                    ids,
                    "cuda",
                    None if qids is None else qids[ids],
                    None if pids is None else pids[ids],
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    seq, done = net.generate(x, seg)
                for i, s, end in zip(ids, seq.tolist(), done.tolist(), strict=True):
                    row = group[i]
                    if ds.rows[i][1:6] != [
                        row[k]
                        for k in ("story", "turn", "question", "history", "answer")
                    ]:
                        raise ValueError("Different dialogue")
                    tokens = s[: s.index(2)] if end else s
                    pred = tok.decode(tokens)
                    em, f1 = metrics(pred, [row["answer"]])
                    row.setdefault("subword", {})[arm] = dict(
                        text=pred,
                        stopped=end,
                        exact=bool(end and em),
                        pixels=end and pred == row["answer"],
                        f1=f1 if end else 0.0,
                    )
        records += group
    out.mkdir(parents=True, exist_ok=False)
    for row in records:
        row["out"] = row.pop("subword")
    (out / "decisions.json.gz").write_bytes(
        gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0)
    )
    print("evaluated", len(records), "turns", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--evaluate-only", action="store_true")
    a = p.parse_args()
    if a.evaluation.exists() or (not a.evaluate_only and a.model.exists()):
        raise FileExistsError("Choose a fresh destination")
    with gpu_owner():
        torch.manual_seed(710491)
        data = read_data()
        if a.evaluate_only:
            from scripts.answer_dialogue_subword import load

            net, tok = load(a.model, "cuda")
        else:
            net, tok = fit(data, a.model)
        evaluate(data, net.eval(), tok, a.evaluation)
