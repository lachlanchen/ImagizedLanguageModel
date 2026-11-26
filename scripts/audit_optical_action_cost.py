"""Count actual padded classification scores in the matched writing schedules."""
from pathlib import Path
from subprocess import run

import numpy as np
from scripts.train_independent_optical_actions import EPOCHS, MODELS, schedule
from scripts.reserve_independent_optical_actions import PUB, ROOT, RUN, dump, read, sha


def main():
    prep = read(PUB/"preparation.json")
    words = np.load(RUN/"cells.npy", mmap_mode="r")
    lens = np.count_nonzero(words,axis=2)
    group = (lens[:,0]//12)*10 + lens[:,1]//12
    tables = read(RUN/"tables.json")
    stats = {}
    for arm in ("cell","strip"):
        fit = read(MODELS/arm/"fit.json")
        assert len(fit["epochs"])==EPOCHS and sha(MODELS/arm/"weights.safetensors")==fit["weights"]
        sizes = [np.count_nonzero(np.load(RUN/f"{arm}_{t}.npy",mmap_mode="r"),axis=1) for t in (0,1)]
        padded = scored = decisions = 0
        for epoch,log in enumerate(fit["epochs"]):
            real = 0
            for indices,t in schedule(group,epoch):
                n = sizes[t][indices]
                count = int(n.sum())+len(n)
                decisions += count
                real += count
                scored += count*len(tables[arm][t])
                padded += (int(n.max())+1)*len(n)*len(tables[arm][t])
            assert real == log["decisions"]
        total_time = sum(v["seconds"] for v in fit["epochs"])
        stats[arm] = dict(decisions=decisions,head_scores=scored,padded_head_scores=padded,
                          seconds=total_time,examples_per_second=2*prep["pairs"]*EPOCHS/total_time,
                          trainable=fit["parameters"],allocated_bytes=max(e["peak_allocated"] for e in fit["epochs"]))
    proof = ROOT/"formal/span_list/VisualActionBudget.lean"
    lean = ROOT/"artifacts/role_binding_v1/tools/lean-4.12.0-linux/bin/lean"
    run([lean,proof],check=True)
    reg = stats["cell"]; alt = stats["strip"]
    crossing = (alt["head_scores"]-reg["head_scores"])/(reg["decisions"]-alt["decisions"])
    dump(PUB/"cost.json",dict(stats=stats,break_even_per_decision=crossing,
          definition="unpadded score-equivalents of shared work, identical in both arms",
          proof=sha(proof),lean=run([lean,"--version"],capture_output=True,text=True,check=True).stdout.strip(),
          preparation=sha(PUB/"preparation.json"),program=sha(Path(__file__))))
    print(stats,flush=True)


if __name__=="__main__":
    main()
