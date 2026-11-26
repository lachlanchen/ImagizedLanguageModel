"""Compare four full passes of diverse or length-matched repeated literary pairs."""

from time import monotonic
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from scripts.sample_visual_library import ROOT, RUN, EVIDENCE, MODELS, RECEIPT, library, schedule, read, sha
from scripts.train_teacher_translation import replay, batches
from scripts.train_visual_prefix import atomic_save, gpu_owner, pack, development, material
from scripts.train_paired_ink import conditional
from scripts.prepare_teacher_translation import dump
from scripts.write_ordered_translation import restore
from scripts.verse_gpu_runtime import configure_cuda_cli


def train(arm, rr, plans, dev, binding):
    path = MODELS / arm
    path.mkdir(exist_ok=True, parents=True)
    fit = path / "fit.json"
    weights = path / "weights.safetensors"
    if fit.exists():
        saved = read(fit)
        if saved["binding"] != binding or saved["weights"] != sha(weights):
            raise ValueError("Changed completed fit")
        return saved
    x = pack([r["input"] for r in rr], "cuda")
    y = pack([r["target"] + [1] for r in rr], "cuda")
    lx = np.asarray([len(r["input"]) for r in rr])
    ly = np.asarray([len(r["target"]) + 1 for r in rr])
    f, inv, maps = optical = material("cuda")
    torch.manual_seed(34105)
    net = restore("prefix-recovery", "cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=0.00016, fused=True)
    history = []
    state = RUN / (arm + "-resume.pt")
    if state.exists():
        record = torch.load(state, map_location="cpu", weights_only=True)
        if record["binding"] != binding:
            raise ValueError("Resume binding differs")
        net.load_state_dict(record["net"])
        opt.load_state_dict(record["opt"])
        history = record["history"]
        del record
    for ep in range(len(history), 4):
        start = monotonic()
        torch.cuda.reset_peak_memory_stats()
        net.train()
        opt.param_groups[0]["lr"] = 0.00016 - ep * 0.00004
        total, tally = 0., 0
        for t, ix in plans[ep]:
            ii = torch.as_tensor(ix, device="cuda")
            n = int(ly[ix].sum())
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                objective = -conditional(net, f, x[ii, :lx[ix].max()], y[ii, :ly[ix].max()], inv[t], maps[t], t).sum() / n
            objective.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            tally += n
            total += float(objective.detach()) * n
        scores = development(net, dev, optical)
        history.append(dict(epoch=ep + 1, nll=total / tally, development=scores, stops=len(rr),
                            target_positions=tally, source_glyphs=int(lx.sum()), updates=len(plans[ep]),
                            seconds=monotonic()-start, peak_bytes=torch.cuda.max_memory_allocated()))
        if tally != ly.sum() or history[-1]["peak_bytes"] > 12 * 2**30 or sum(h["seconds"] for h in history) > 3600:
            raise ValueError("Training accounting or resource limit")
        atomic_save(lambda p: torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), history=history, binding=binding), p), state)
        print(arm, history[-1], flush=True)
    atomic_save(lambda p: save_file({k: v.detach().cpu() for k,v in net.state_dict().items()}, p), weights)
    result = dict(weights=sha(weights), binding=binding, history=history)
    dump(fit, result)
    return result


def main():
    configure_cuda_cli()
    for p in (RUN, EVIDENCE):
        p.mkdir(exist_ok=True, parents=True)
    rows = library("train")
    ix, sampling = schedule(rows)
    dump(EVIDENCE / "sampling.json", sampling)
    common = replay()
    arms = {}
    for arm in ("broad", "repeat"):
        arr = []
        for t in (0, 1):
            for i in range(len(rows)):
                pair = rows[i if arm == "broad" else ix[t][i]]["cells"]
                arr.append(dict(input=pair[1-t], target=pair[t], t=t))
        arms[arm] = arr + common
    if any(len(a["target"]) != len(b["target"]) or a["t"] != b["t"] for a,b in zip(*arms.values(),strict=True)):
        raise ValueError("Unmatched target exposure")
    plans = [batches(arms["broad"], ep) for ep in range(4)]
    binding = {str(p.relative_to(ROOT)):sha(p) for p in
               (Path(__file__).resolve(), ROOT/"scripts/sample_visual_library.py", RECEIPT, EVIDENCE/"sampling.json",
                ROOT/"references/visual_library_diversity_protocol.md", ROOT/"models/visual_prefix/recovery/weights.safetensors")}
    dev = [dict(r, domain=r["work"]) for r in library("dev")]
    with gpu_owner():
        if torch.cuda.mem_get_info()[0] < 7 * 2**30:
            raise RuntimeError("Insufficient free GPU memory")
        fits = [train(k, arms[k], plans, dev, binding) for k in ("repeat","broad")]
    for a,b in zip(*(s["history"] for s in fits),strict=True):
        if any(a[k] != b[k] for k in ("updates","source_glyphs","target_positions","stops")):
            raise ValueError("Unequal matched budget")


if __name__ == "__main__":
    main()
