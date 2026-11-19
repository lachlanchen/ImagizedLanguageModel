"""Train the physical span decision and evaluate complete answers with frozen experts."""
from collections import Counter
import fcntl
import json
import math
import os
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file

from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.pictured_answer_selection import AnswerSelection, read_options
from scripts.evaluate_pointing_causal_writer import READER, load_streams, metrics, store_streams
from scripts.pointing_glyph_corpus import Row
from scripts.prepare_composed_chinese_answers import Rasterizer
from scripts.train_pointing_causal_writer import HELD, PUBLIC, ROOT, TRAIN, WORK, digest, load, record, sha


def features(reader, rows):
    xs, quotes = [], []
    for i in range(0, len(rows), 64):
        x, y = read_options(reader, [(r.p, r.q) for r in rows[i:i+64]])
        xs.append(x)
        quotes.extend(y)
    return torch.cat(xs), quotes


def fit(x, labels, shuffled):
    torch.manual_seed(18170)
    net = AnswerSelection()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0004)
    batches = torch.Generator().manual_seed(18171)
    if shuffled:
        labels = labels[torch.randperm(len(x), generator=torch.Generator().manual_seed(18172))]
    for _ in range(8):
        for ids in torch.randperm(len(x), generator=batches).split(64):
            logits = net(x[ids]).flatten()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return net.eval()


def main():
    t = monotonic()
    train, dev = load()
    assert digest(train) == TRAIN and digest(dev) == HELD
    reader = FontInvariantReader.restore(READER, "cuda")
    nreader = sum(p.numel() for p in reader.parameters())
    tx, _ = features(reader, train)
    labels = torch.tensor([any(r.p[i:i+len(r.y)] == r.y for i in range(0, len(r.p), 128)) for r in train]).float()
    arms = {name: fit(tx, labels, name == "shuffled") for name in ("learned", "shuffled")}
    for name, model in arms.items():
        save_file(model.state_dict(), str(WORK/(name+"_selection.safetensors")))
    result = dict(training=len(train), positive=int(labels.sum()), updates_per_arm=8*math.ceil(len(train)/64),
                  reader_parameters=nreader, selector_parameters=sum(p.numel() for p in arms["learned"].parameters()),
                  training_sha256=TRAIN, weights=sha(WORK/"learned_selection.safetensors"))
    corpus = json.loads((WORK/"windows.json").read_text())["fresh"]
    raster = Rasterizer()
    def render(text):
        return b"".join(raster.tile(c) for c in text)
    fresh = [Row("fresh", a["key"], render(a["passages"][0]), render(a["question"]), render(a["target"]), b"") for a in corpus]
    info = json.loads((WORK/"fresh_ranked_cases.json").read_text())
    original = {i:k for k,(i,j) in enumerate(info["stream_index"]) if j==0}
    fs, ft = load_streams(WORK/"fresh_ranked_streams.npz")
    assert sha(WORK/"pointing_answers.npz")==json.loads((PUBLIC/"scores.json").read_text())["pointing"]["stream_sha256"]
    for name,rows,(written,stopped) in (("development",dev,load_streams(WORK/"pointing_answers.npz")),
                                        ("fresh",fresh,([fs[original[i]] for i in range(len(fresh))],[ft[original[i]] for i in range(len(fresh))]))):
        x, quotes = features(reader, rows)
        correct = [s == r.y for r,s in zip(rows,quotes,strict=True)]
        with torch.no_grad():
            choices = {k: m(x).flatten().ge(0).tolist() for k,m in arms.items()}
        choices.update(quote=[True]*len(rows), write=[False]*len(rows), reference_assisted=correct)
        if name=="development":
            old,_=load_streams(WORK/"extractive_answers.npz")
            assert old==[q for r,q in zip(rows,quotes) if r.task in ("cmrc","drcd")]
        result[name]={}
        for arm,use in choices.items():
            streams=[q if u else w for q,w,u in zip(quotes,written,use,strict=True)]
            stops=[u or s for u,s in zip(use,stopped,strict=True)]
            out=metrics(rows,streams,stops)
            totals=Counter(r.task for r,u in zip(rows,use) if u)
            for group,v in out.items():v["quoted"]=totals[group]
            if name=="fresh":
                out["fresh"]["multi_exact"]=sum(s and a in [render(v) for v in sample["refs"]] for s,a,sample in zip(stops,streams,corpus))
            file=WORK/f"{name}_{arm}_selected.npz"
            store_streams(file,streams,stops)
            result[name][arm]={"scores":out,"stream_sha256":sha(file)}
        print(name,{k:(v["exact"],v.get("both")) for k,v in result[name]["learned"]["scores"].items()},flush=True)
    result.update(seconds=monotonic()-t, peak_bytes=torch.cuda.max_memory_allocated())
    record(PUBLIC/"selection.json",result)


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        main()
