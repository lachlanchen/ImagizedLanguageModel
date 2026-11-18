"""Test sentence search and known-history drawing with the frozen physical planner."""
from collections import Counter
import fcntl
import resource
import time
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.glyph_planner import GlyphPlanner
from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.glyph_planner_search import search
from scripts.train_glyph_planner import (
    OUT, ROOT, corpora, guard, load, load_split, local_counts, matching_pairs, paired_interval,
    require, sha, summarize, verified_split, write_new,
)

DEST = OUT / "search"


@torch.inference_mode()
def main():
    start = time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    require("4090" in torch.cuda.get_device_name(), "Expected registered GPU")
    baseline = load(OUT/"evaluation.json")
    require(not baseline["passed"], "Expected failed local decoding")
    test = verified_split(OUT/"fresh")
    train, combined = corpora()
    freq = Counter(map(bytes,train["target"]))
    pair = matching_pairs(test["context"],test["target"],test["group"])
    bank = np.load(OUT/"fresh/pixels.npy")
    pictures = torch.from_numpy(np.unpackbits(bank, axis=1)).cuda().bool()
    x, y = [torch.as_tensor(test[k].astype(np.int32), device="cuda") for k in ("context","target")]
    original = {}
    for name, metrics in baseline["arms"].items():
        path = OUT/"fresh"/name/"generated.npz"
        require(sha(path) == metrics["archive_sha256"], "Existing stream changed")
        original[name] = load_split(path)
    DEST.mkdir()
    reports, hits = {}, {}
    for arm in ("shuffled", "aligned"):
        weights = OUT/arm/"weights.pt"
        require(sha(weights) == load(OUT/arm/"fit.json")["weights_sha256"], "Changed final weights")
        state = torch.load(weights, weights_only=True, map_location="cpu")
        net = GlyphPlanner(state["landmarks"]).cuda().eval()
        net.load_state_dict(state)
        features = visual_input(pictures, net.landmarks)
        feedback = torch.cat([net.motor(v[:,net.landmarks])[:,net.landmarks] > 0 for v in pictures.split(128)])
        saved = original[arm]
        chunks, teacher = [], []
        for offset in range(0, len(x), 128):
            guard(start)
            window = slice(offset,offset+128)
            ctx = net.read(features[x[window]])
            if arm == "aligned":
                replay = search(net, ctx, 1)
                for array, key in zip(replay, ("packed","length","stopped")):
                    require(np.array_equal(array, saved[key][window]), f"Width-one {key} changed at {offset}")
                past = net.advance(ctx, ctx, feedback[y[window]].float())
                plans = torch.cat((ctx[:,None], past[:,:-1]), 1)
                cells = torch.stack([net.draw(plans[:,slot].contiguous(),4) for slot in range(4)], 1)
                require(np.array_equal(np.packbits(cells[:,0].cpu().numpy(),axis=-1), saved["packed"][window,0]),
                        "Supplied-history first drawing differs")
                live = net.stop(plans).squeeze(-1) < 0
                correct = (cells == pictures[y[window]]).all(-1) & live
                teacher.append(correct.cpu().numpy())
            chunks.append(search(net,ctx,4))
        packed, length, stopped = (np.concatenate(v) for v in zip(*chunks))
        archive = dict(packed=packed, length=length, stopped=stopped)
        path = DEST/(arm+".npz")
        np.savez_compressed(path, **archive)
        hit, row = summarize(archive,test,bank,combined,freq,pair)
        correct = (packed[:,:4] == bank[test["target"]]).all(-1) & (np.arange(4) < length[:,None])
        valid = np.arange(8) < length[:,None]
        known = set(map(bytes,bank[np.unique(combined["target"])]))
        row.update(prefix_exact=np.logical_and.accumulate(correct,axis=1).sum(0).tolist(),
                   position_exact=correct.sum(0).tolist(), emitted=int(valid.sum()),
                   known=sum(v.tobytes() in known for v in packed[valid]), archive_sha256=sha(path))
        if teacher:
            supplied = np.concatenate(teacher)
            np.savez_compressed(DEST/"supplied_history.npz", exact=supplied)
            free = (saved["packed"][:,:4] == bank[test["target"]]).all(-1) & (np.arange(4)<saved["length"][:,None])
            previous_ok = np.column_stack((np.ones(len(free),bool), np.logical_and.accumulate(free,axis=1)[:,:-1]))
            row.update(supplied_position_exact=supplied.sum(0).tolist(),
                       supplied_all_four=int(supplied.all(1).sum()), replayed=len(free),
                       supplied_after_wrong_prefix=(supplied & ~previous_ok).sum(0).tolist(),
                       agreement_after_correct_prefix=int(((supplied==free)|~previous_ok).all(1).sum()),
                       supplied_sha256=sha(DEST/"supplied_history.npz"))
        reports[arm], hits[arm] = row, hit
        print(arm, row, flush=True)
    for key in ("aligned", "uniform", "released"):
        hits[key+"_local"] = summarize(original[key],test,bank,combined,freq,pair)[0]
    hits["counts"] = local_counts(combined,test)
    a = hits["aligned"]
    comparison = {name:dict(gained=int((a&~other).sum()), lost=int((~a&other).sum()),
                            interval_pp=paired_interval(a,other,test["group"])) for name,other in hits.items() if name!="aligned"}
    guard(start)
    write_new(DEST/"result.json", dict(arms=reports, comparisons=comparison, rows=len(test["group"]),
              seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, program_sha256=sha(Path(__file__)),
              protocol_sha256=sha(ROOT/"references/glyph_planner_search_protocol.md")))


if __name__ == "__main__":
    with (OUT.parent/"ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
