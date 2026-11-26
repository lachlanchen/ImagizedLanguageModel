"""Matched masking and displacement training of bilingual visual-strip writers."""
import argparse
from hashlib import sha256
import math
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional as F

from ilm.visual_lm.optical_actions import OpticalActions, pool_tiles, strip_indices
from scripts.prepare_independent_optical_actions import PUB as CORPUS_PUB, RUN as CORPUS
from scripts.reserve_visual_strip_restoration import BASE, PROTOCOL, PUB, ROOT, RUN, dump, read, sha
from ilm.visual_lm.strip_restoration import RestoringWriter, corrupt
from scripts.train_independent_optical_actions import schedule

ARMS = ("mask", "restore")
from scripts.train_field_to_glyph import atomic_save, configure_cuda_cli, gpu_owner

MODELS = ROOT / "models/visual_strip_restoration"
EPOCHS = 4
BATCH = 192
SEED = 29377


def training(arm):
    prep = read(CORPUS_PUB / "preparation.json")
    assert sha(RUN/"panel.json") == read(PUB/"panel.json")["sha256"]
    assert all(sha(CORPUS/name) == digest for name, digest in prep["files"].items())
    arch = ROOT / "ilm/visual_lm/optical_actions.py"
    binding = dict(preparation=sha(CORPUS_PUB/"preparation.json"), panel=sha(PUB/"panel.json"),
                   protocol=sha(PROTOCOL), training=sha(Path(__file__)), architecture=sha(arch), new_body=sha(ROOT/"ilm/visual_lm/strip_restoration.py"),
                   schedule=sha(ROOT/"scripts/train_independent_optical_actions.py"),
                   parent=sha(ROOT/"models/independent_optical_actions/strip/weights.safetensors"),
                   blocks=sha(ROOT/"ilm/visual_lm/ordered_translation.py"), seed=SEED)
    out = MODELS / arm
    out.mkdir(parents=True, exist_ok=True)
    result = out / "fit.json"
    weights = out / "weights.safetensors"
    if result.exists():
        rec = read(result)
        assert rec["binding"] == binding and rec["weights"] == sha(weights)
        return
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    net = RestoringWriter().cuda()
    path = ROOT/"models/independent_optical_actions/strip"
    assert binding["parent"] == read(path/"fit.json")["weights"]
    missing, extra = net.load_state_dict(load_file(str(path/"weights.safetensors")), strict=False)
    assert set(missing)=={"mask_form", "length_head.0.weight", "length_head.0.bias", "length_head.2.weight", "length_head.2.bias"} and not extra
    opt = torch.optim.AdamW(net.parameters(), lr=.0005, weight_decay=.01, fused=True)
    raw = np.load(CORPUS/"cells.npy")
    lens = np.count_nonzero(raw, axis=2)
    group = (lens[:,0]//12)*10 + lens[:,1]//12
    train = torch.tensor(raw.astype(np.int32), device="cuda")
    tiles = pool_tiles(torch.from_numpy(np.unpackbits(np.load(BASE/"palette.npy"), axis=1)
                            .reshape(-1,32,32)).cuda())
    targets = [np.load(CORPUS/f"strip_{t}.npy") for t in (0,1)]
    sizes = [np.count_nonzero(x, axis=1) for x in targets]
    target = [torch.tensor(x.astype(np.int32), device="cuda") for x in targets]
    tables = [strip_indices(a, "cpu").cuda() for a in read(CORPUS/"tables.json")["strip"]]
    resume = RUN/f"{arm}_resume.pt"
    records = []
    if resume.exists():
        checkpoint = torch.load(resume, map_location="cuda", weights_only=True)
        assert checkpoint["binding"] == binding
        net.load_state_dict(checkpoint["net"])
        opt.load_state_dict(checkpoint["opt"])
        records = checkpoint["epochs"]
        del checkpoint
    start = monotonic()
    prior = sum(e["seconds"] for e in records)
    total_steps = EPOCHS * len(schedule(group, 0))
    step = sum(e["updates"] for e in records)
    for epoch in range(len(records), EPOCHS):
        rng = torch.Generator(device="cuda").manual_seed(SEED + epoch)
        clock = monotonic()
        examples = decisions = 0
        nll = torch.zeros((), device="cuda", dtype=torch.float64)
        order_hash = sha256()
        batches = schedule(group, epoch)
        for j, (idx,t) in enumerate(batches):
            order_hash.update(bytes([t]) + idx.astype("<i4").tobytes())
            step += 1
            rate = min(1., step/200) * (.15 + .85 * (1+math.cos(math.pi*step/total_steps))/2)
            opt.param_groups[0]["lr"] = .0005 * rate
            ix = torch.from_numpy(idx).cuda()
            w, h = int(lens[idx,1-t].max()), int(sizes[t][idx].max())
            src = train[ix,1-t,:w].long()
            gold = target[t][ix,:h].long()
            length = gold.ne(0).sum(1)
            observed, selected = corrupt(gold, arm == "restore", rng)
            labels = (gold-1).masked_fill(~selected, -100)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits, length_logits = net(tiles[src],src.ne(0).sum(1),observed,gold.ne(0),tiles,tables[t],t)
                content_loss = F.cross_entropy(logits.flatten(0,1),labels.flatten())
                loss = content_loss + .35*F.cross_entropy(length_logits,length-1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            count = int(selected.sum())
            nll += content_loss.detach()*count
            decisions += count
            examples += len(idx)
            if j % 250 == 0:
                print(arm,epoch+1,j,len(batches),round(loss.item(),3),round(monotonic()-clock,1),flush=True)
                if prior + monotonic()-start > 10800 or torch.cuda.max_memory_allocated() > 20*2**30:
                    raise RuntimeError("Registered resource ceiling reached")
        assert examples == 2*len(raw)
        records.append(dict(epoch=epoch+1,updates=len(batches),examples=examples,decisions=decisions,
                            nll_per_action=nll.item()/decisions,seconds=monotonic()-clock,
                            order=order_hash.hexdigest(),peak_allocated=torch.cuda.max_memory_allocated()))
        atomic_save(lambda path: torch.save(dict(net=net.state_dict(),opt=opt.state_dict(),
                    epochs=records,binding=binding),path),resume)
        dump(RUN/f"{arm}_progress.json",records)
    atomic_save(lambda path: save_file({k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()},path),weights)
    dump(result,dict(arm=arm,binding=binding,epochs=records,weights=sha(weights),
                     parameters=sum(p.numel() for p in net.parameters()),gpu=torch.cuda.get_device_name()))
    print("Finished", arm, flush=True)


def main():
    configure_cuda_cli()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(*ARMS,"both"), default="both")
    arm = parser.parse_args().arm
    with gpu_owner():
        for choice in (ARMS if arm=="both" else [arm]):
            torch.cuda.reset_peak_memory_stats()
            training(choice)


if __name__ == "__main__":
    main()
