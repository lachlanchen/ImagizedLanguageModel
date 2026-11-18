"""Train matched visual planners and independently score stopped, emitted pixels."""
import argparse
from collections import Counter
import fcntl
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.glyph_planner import GlyphPlanner
from ilm.visual_lm.idiom_ink import GROUPS, STOP, nibbles
from scripts.compare_passage_diversity import (
    DATA, FRESH, HINTS, JOINT, PREVIOUS, ROOT, OUT as DIVERSITY, corpora, guard, infer, integrated_check,
    load, load_initial, load_split, local_counts, matching_pairs, paired_interval, physical_bank, prepare,
    require, sha, summarize, verified_split, write_new,
)
from scripts.train_ordered_neighborhood import OUT as NEIGHBOR, WEIGHT

OUT = ROOT / "artifacts/glyph_planner_v1"
PROTOCOL = ROOT / "references/glyph_planner_protocol.md"


@torch.no_grad()
def verify_trajectory(net, pictures, bits, feedback):
    net.eval()
    batched = net.trajectory_loss(pictures, bits, feedback)
    context = net.read(pictures)
    state = context
    total = context.new_zeros(())
    for slot in range(bits.shape[1]):
        total += F.softplus(net.stop(state)).sum()
        prefix = torch.zeros_like(bits[:, slot])
        previous = torch.full((len(bits),), 17, dtype=torch.long, device=bits.device)
        hidden = state
        labels = nibbles(bits[:, slot])
        for group in range(GROUPS):
            inputs = net.inputs(previous, torch.full_like(previous, group), prefix, state)
            step, _ = net.causal(inputs[:, None], hidden[None].contiguous())
            hidden = step[:, 0]
            previous = labels[:, group]
            total += F.cross_entropy(net.next_pattern(hidden)[:, :STOP], previous, reduction="sum")
            prefix[:, 4*group:4*group+4] = bits[:, slot, 4*group:4*group+4]
        state = net.advance(context, state, feedback[:, slot:slot+1])[:, 0]
    total += F.softplus(-net.stop(state)).sum()
    stepwise = total / (len(bits) * (bits.shape[1]*(GROUPS+1) + 1))
    torch.testing.assert_close(batched, stepwise, atol=2e-5, rtol=2e-5)
    stop_weights = {k: v.clone() for k, v in net.stop.state_dict().items()}
    net.stop.weight.zero_()
    for bias, expected_len, expected_stop in [(10, 0, True), (-10, 2, False)]:
        net.stop.bias.fill_(bias)
        pixels, length, stopped = net.generate(pictures, limit=2)
        require((length == expected_len).all() and (stopped == expected_stop).all(), "Stop/limit mismatch")
        require(expected_len != 0 or not pixels.any(), "Unexpected ink after immediate stop")
    net.stop.load_state_dict(stop_weights)
    return dict(batched_nll=batched.item(), stepwise_nll=stepwise.item(), stop_and_limit_verified=True)


def train(start):
    initial = load_initial()
    _, train_data = corpora()
    x, y = [torch.as_tensor(train_data[k].astype(np.int32), device="cuda") for k in ("context", "target")]
    require(len(y) == 142878, "Training population differs")
    features, bits, feedback, _ = physical_bank(initial, np.load(DIVERSITY / "pool/pixels.npy"))
    torch.manual_seed(720)
    prototype = GlyphPlanner(initial["landmarks"])
    missing, unused = prototype.load_state_dict(initial, strict=False)
    require(not unused and set(missing) == {k for k in prototype.state_dict() if k.startswith(("planner.", "stop."))},
            "Incomplete inherited weights")
    OUT.mkdir()
    for arm in ("shuffled", "aligned"):
        torch.manual_seed(720)
        model = GlyphPlanner(initial["landmarks"]).cuda()
        model.load_state_dict(prototype.state_dict())
        check = verify_trajectory(model, features[x[:8]], bits[y[:8]], feedback[y[:8]])
        model.train()
        model.reader.eval()  # Disable dropout but propagate passage gradients in both arms.
        model.motor.requires_grad_(False)
        trainable = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=.0002, weight_decay=.01)
        rng = np.random.default_rng(720)
        history = []
        begin = time.monotonic()
        for epoch in range(4):
            order = rng.permutation(len(y))
            donors = np.roll(order, len(order)//2) if arm == "shuffled" else order
            require(arm != "shuffled" or np.all(donors != order), "Shuffled passages must be distinct")
            nll = 0.
            for pos in range(0, len(y), 192):
                if pos % 3072 == 0:
                    guard(start)
                target = y[order[pos:pos+192]]
                source = x[donors[pos:pos+192]]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model.trajectory_loss(features[source], bits[target], feedback[target])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                optimizer.step()
                nll += loss.item() * len(target)
            history.append(nll / len(y))
            print(arm, epoch+1, history[-1], flush=True)
        path = OUT / arm
        path.mkdir()
        weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        require(all(torch.equal(weights[k], v) for k, v in initial.items()
                    if k.startswith("motor.") or k == "landmarks"), "Renderer changed")
        torch.save(weights, path / "weights.pt")
        write_new(path / "fit.json", dict(history=history, weights_sha256=sha(path/"weights.pt"),
                  examples_seen=4*len(y), updates=4*((len(y)+191)//192), fit_seconds=time.monotonic()-begin,
                  total=sum(v.numel() for v in model.parameters()), trainable=sum(v.numel() for v in trainable),
                  preflight=check))


def evaluate(start):
    original, combined = corpora()
    folder = OUT / "fresh"
    folder.mkdir()
    earlier = (FRESH, HINTS/"confirmation", JOINT/"confirmation", PREVIOUS, DIVERSITY/"fresh",
               DIVERSITY/"pool", WEIGHT/"fresh", NEIGHBOR/"fresh")
    test, bank = prepare(original, load_split(DATA/"validation.npz"), np.load(NEIGHBOR/"fresh/pixels.npy"),
                         buckets=range(124,132), destination=folder, excluded=[verified_split(p) for p in earlier])
    frequency = Counter(map(bytes, original["target"]))
    pairs = matching_pairs(test["context"], test["target"], test["group"])
    hits = {"local_counts": local_counts(combined, test)}
    reports = {}
    for arm, model in (("released", JOINT/"reference"), ("uniform", DIVERSITY/"diverse"),
                       ("shuffled", OUT/"shuffled"), ("aligned", OUT/"aligned")):
        path = folder / arm
        kwargs = {"model_type": GlyphPlanner} if arm in ("aligned", "shuffled") else {}
        infer(model/"weights.pt", path, combined, test, bank, start, **kwargs)
        stream = load_split(path/"generated.npz")
        hit, row = summarize(stream, test, bank, combined, frequency, pairs)
        metrics = load(path/"result.json")
        require(row["exact"] == metrics["complete_exact"], "Pixel recount disagrees")
        present = np.arange(4) < stream["length"][:, None]
        glyphs = (stream["packed"][:, :4] == bank[test["target"]]).all(-1) & present
        reports[arm] = dict(row, position_exact=glyphs.sum(0).tolist(),
                            prefix_exact=np.logical_and.accumulate(glyphs, axis=1).sum(0).tolist(),
                            swapped_exact=metrics["swapped_exact"], lengths=metrics["lengths"],
                            emitted=metrics["emitted_cells"], known=metrics["training_target_shape_hits"],
                            local_misses_correct=int((hit & ~hits["local_counts"]).sum()),
                            archive_sha256=sha(path/"generated.npz"))
        hits[arm] = hit
        print(arm, row["exact"], reports[arm]["prefix_exact"], flush=True)
    candidate = hits["aligned"]
    comparisons = {name: dict(gained=int((candidate & ~other).sum()), lost=int((~candidate & other).sum()),
                              interval_pp=paired_interval(candidate,other,test["group"]))
                   for name,other in hits.items() if name != "aligned"}
    a = reports["aligned"]
    passed = (all(v["interval_pp"][0] > 0 for v in comparisons.values()) and a["known"] >= .99*a["emitted"]
              and a["both_correct"] >= reports["uniform"]["both_correct"])
    result = dict(rows=len(candidate), groups=len(np.unique(test["group"])), local_exact=int(hits["local_counts"].sum()),
                  arms=reports, comparisons=comparisons, passed=bool(passed))
    if passed:
        weights = torch.load(OUT/"aligned/weights.pt", weights_only=True, map_location="cpu")
        model = GlyphPlanner(weights["landmarks"]).cuda().eval()
        model.load_state_dict(weights)
        result["historical"] = integrated_check(model, test, bank, start, generated=folder/"aligned/generated.npz")
    write_new(OUT/"evaluation.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    phase = parser.parse_args().phase
    with (OUT.parent/"ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        guard(start)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        require("4090" in torch.cuda.get_device_name(), "Expected registered GPU")
        (train if phase == "train" else evaluate)(start)
        guard(start)
        write_new(OUT/(phase+"_resources.json"), dict(seconds=time.monotonic()-start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  device=torch.cuda.get_device_name(), program_sha256=sha(Path(__file__)),
                  architecture_sha256=sha(ROOT/"ilm/visual_lm/glyph_planner.py"), protocol_sha256=sha(PROTOCOL)))
