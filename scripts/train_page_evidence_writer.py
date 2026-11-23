"""Train matched causal image writers with full or compressed pictured evidence."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch

from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter
from scripts.page_evidence_corpus import DEFAULT, SEED, make_splits, save_split
from scripts.pointing_glyph_corpus import Row
from scripts.prepare_visual_facts import Face, FONTS, LANG, PUBLIC, WORK
from scripts.train_causal_glyph_writer import in_cells, record, sha
from scripts.train_pointing_causal_writer import inputs, objective
from scripts.train_span_choice import gpu_owner

BASE = WORK / "writer"


def source(v, arm):
    return v["p"] if arm == "page" else "　".join((v["s"], LANG[v["r"]][0], v["o"]))


def writable_records(face, net, raw):
    fixed = {row.tobytes() for row in net.writer.packed.cpu().numpy()}
    keep, omitted = [], Counter()
    for v in raw:
        missing = {face.cell(c) for c in v["y"]} - fixed
        if all(missing <= {face.cell(c) for c in source(v, arm)} for arm in ("page", "fields")):
            keep.append(v)
        else:
            omitted[v["task"]] += 1
    return keep, dict(omitted)


def fit(folder, arm, data, face):
    torch.manual_seed(SEED)
    net = PointingCausalWriter.restore(BASE, "cuda")
    net.writer.requires_grad_(False)
    net.writer.body.h[-4:].requires_grad_(True)
    net.writer.body.ln_f.requires_grad_(True)
    optim = torch.optim.AdamW([
        {"params": [p for p in net.writer.parameters() if p.requires_grad], "lr": 5e-5},
        {"params": net.pointer.parameters(), "lr": 4e-4},
    ], fused=True)
    material = [Row(v["task"], v["key"], face(source(v, arm)), face(v["q"]), face(v["y"]), b"") for v in data]
    read, alphabet, pairs, answers = inputs(net.writer, material)
    del material
    rng = np.random.default_rng(SEED)
    updates = positions = 0
    losses = []
    torch.cuda.reset_peak_memory_stats()
    t = monotonic()
    for epoch in range(2):
        total = count = 0
        for pool in np.array_split(rng.permutation(len(data)), max(1, len(data)//512)):
            ordered = sorted(pool, key=lambda i: len(data[i]["p"]) + 2 * len(data[i]["q"]))
            for ids in np.array_split(ordered, int(np.ceil(len(ordered)/20))):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, nll, n = objective(net, read, alphabet, [pairs[i] for i in ids],
                                             [answers[i] for i in ids], False)
                if not torch.isfinite(loss) or loss.detach().item() > 100:
                    raise ValueError("Invalid physical writing loss")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
                optim.step()
                updates += 1
                total += float(nll)
                count += n
                if torch.cuda.max_memory_allocated() > 20 * 2**30 or monotonic()-t > 1200:
                    raise RuntimeError("Writer training budget exceeded")
                if updates % 250 == 0:
                    print(arm, epoch+1, updates, round(total/count, 3), flush=True)
        losses.append(total/count)
        positions += count
    elapsed = monotonic()-t
    temp = folder / (arm + ".partial")
    temp.mkdir()
    for f in ("config.json", "interface.safetensors", "metric.safetensors"):
        try:
            os.link(BASE/f, temp/f)
        except OSError:
            import shutil
            shutil.copyfile(BASE/f, temp/f)
    for name, module in (("body", net.writer.body), ("pointer", net.pointer)):
        save_file({k: v.detach().cpu().contiguous() for k,v in module.state_dict().items()},
                  str(temp/(name+".safetensors")))
    temp.rename(folder/arm)
    out = dict(rows=len(data), epochs=losses, updates=updates, exposures=positions,
               optimization_seconds=elapsed, peak_bytes=torch.cuda.max_memory_allocated(),
               trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),
               parameters=sum(p.numel() for p in net.parameters()),
               weights={p.name:sha(p) for p in (folder/arm).iterdir()})
    record(folder/(arm+"_fit.json"), out)
    print(arm, "finished", round(elapsed,1), losses, flush=True)


def train(folder):
    folder.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(2)
    t = monotonic()
    original = json.loads((PUBLIC/"writer_fit.json").read_text())["weights"]
    if any(sha(BASE/k)!=h for k,h in original.items()):
        raise ValueError("Writer initialization changed")
    face = Face((FONTS[0], 2))
    data, held = make_splits()
    base = PointingCausalWriter.restore(BASE, "cpu")
    data, omitted = writable_records(face, base, data)
    del base
    manifest = dict(training=save_split(folder,"training",data), held=save_split(folder,"held",held),
                    training_omitted_by_support=omitted, base=original, seed=SEED,
                    preparation_seconds=monotonic()-t)
    record(folder/"corpus.json", manifest)
    print(manifest["training"], manifest["held"], flush=True)
    for arm in ("page", "fields"):
        fit(folder, arm, data, face)
    record(folder/"training.json", dict(gpu=torch.cuda.get_device_name(), seconds=monotonic()-t))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT)
    with gpu_owner():
        train(parser.parse_args().run_dir)
