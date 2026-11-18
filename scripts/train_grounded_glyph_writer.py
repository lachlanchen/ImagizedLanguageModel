"""Train and compare independent pictured-question writers against a broken-pairing control."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from ilm.visual_lm.composed_answer_writer import descriptors
from ilm.visual_lm.grounded_glyph_decoder import GroundedGlyphDecoder
from scripts.audit_composed_writer_outputs import distance
from scripts.prepare_grounded_answer_images import ANSWERS, QUESTIONS, DEST
from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.train_composed_answer_writer import DATA, ROOT, Split, padded

WORK = ROOT / "artifacts/grounded_glyph_writer_v1"
REG = ROOT / "references/evidence/grounded_glyph_writer_registration.json"
SEED, BATCH, LIMIT = 19427, 96, 64
PHASES = (("large", 1, .001), ("adapt", 6, .00025))


def bindings():
    files = [Path(__file__), ROOT / "ilm/visual_lm/grounded_glyph_decoder.py",
             ROOT / "scripts/prepare_grounded_answer_images.py",
             ROOT / "references/grounded_glyph_writer_protocol.md", DATA / "preparation.json",
             *(folder / "receipt.json" for folder in (ANSWERS, QUESTIONS, DEST))]
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


class ShapeBank:
    """Zero denotes padding or termination. A genuine white tile gets a positive address."""
    def __init__(self, base):
        self.tiles = list(map(bytes, base))
        self.lookup = {b: i for i, b in enumerate(self.tiles) if i}
        assert len(self.lookup) == len(self.tiles) - 1

    def merge(self, tiles):
        remap = []
        for tile in map(bytes, tiles):
            if tile not in self.lookup:
                self.lookup[tile] = len(self.tiles)
                self.tiles.append(tile)
            remap.append(self.lookup[tile])
        return np.asarray(remap)

    def features(self, device):
        bits = np.unpackbits(np.frombuffer(b"".join(self.tiles), np.uint8).reshape(-1, 128), axis=1)
        return descriptors(torch.as_tensor(bits, device=device))


class Triples:
    def __init__(self, p, q, y):
        self.p, self.q, self.y = p, q, y
        self.lengths = np.array(list(map(len, y)))
        assert len(p) == len(q) == len(y)
        assert min(self.lengths) and min(map(len, p)) and min(map(len, q))

    def inputs(self, indices, qidx, device):
        result = []
        for seqs, ix in ((self.p, indices), (self.q, qidx)):
            result.append(torch.as_tensor(padded([seqs[i] for i in ix]), device=device))
        return result

    def targets(self, indices, device):
        return torch.as_tensor(padded([self.y[i] for i in indices]), device=device)


def split_storage(seq, offsets):
    assert offsets[0] == 0 and offsets[-1] == len(seq) and np.all(np.diff(offsets) > 0)
    return np.split(seq, offsets[1:-1])


def read_large():
    needed = {ANSWERS: ("sequence.npy", "lengths.npy", "tiles.npy"),
              QUESTIONS: ("questions.npy", "offsets.npy", "tiles.npy", "answer_rows.npy"),
              DEST: ("source_handles.bin", "rows.npy", "offsets.npy", "tiles.npy")}
    for directory, files in needed.items():
        expected = load(directory / "receipt.json")["files"]
        for name in files:
            assert sha(directory / name) == expected[name]
    base, qt, at = [np.load(p / "tiles.npy") for p in (DEST, QUESTIONS, ANSWERS)]
    assert np.array_equal(base[:len(qt)], qt) and np.array_equal(qt[:len(at)], at)
    rows = np.load(DEST / "rows.npy")
    assert np.array_equal(np.load(QUESTIONS / "answer_rows.npy"), np.arange(192230))
    assert len(rows) == 191007 and np.all(np.diff(rows) > 0)
    passages = split_storage(np.memmap(DEST / "source_handles.bin", dtype=np.uint32, mode="r"),
                             np.load(DEST / "offsets.npy"))
    questions = split_storage(np.load(QUESTIONS / "questions.npy"), np.load(QUESTIONS / "offsets.npy"))
    answers = split_storage(np.load(ANSWERS / "sequence.npy"),
                            np.r_[0, np.cumsum(np.load(ANSWERS / "lengths.npy"))])
    return ShapeBank(base), Triples(passages, [questions[i] for i in rows], [answers[i] for i in rows])


def small(bank, split):
    ds = Split(split)
    remap = bank.merge(ds.tiles)
    return Triples(*[[remap[r] for r in seqs] for seqs in (ds.p, ds.q, ds.y)]), ds.groups


def order_for(lengths, epoch):
    rng = np.random.default_rng(SEED + epoch)
    order = rng.permutation(len(lengths))
    # Shuffle buckets, then reduce padding within each group of sixteen minibatches.
    for block in range(0, len(order), BATCH * 16):
        local = order[block:block + BATCH * 16]
        local = local[np.argsort(lengths[local], kind="stable")]
        batches = [local[j:j + BATCH] for j in range(0, len(local), BATCH)]
        for j in rng.permutation(len(batches)):
            yield batches[j]


def question_order(size, broken):
    ix = np.arange(size)
    if broken:
        shuffled = np.random.default_rng(SEED + size).permutation(size)
        ix[shuffled] = np.roll(shuffled, 1)
        assert np.all(ix != np.arange(size))
    return ix


def fit(arm, splits, features, memory, ids, device):
    directory = WORK / arm
    directory.mkdir()
    torch.manual_seed(SEED)
    net = GroundedGlyphDecoder().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    rows, total_steps, started = [], 0, monotonic()
    torch.cuda.reset_peak_memory_stats()
    for phase, epochs, lr in PHASES:
        data = splits[phase]
        given = question_order(len(data.y), arm == "deranged")
        for epoch in range(epochs):
            totals = torch.zeros(2, device=device, dtype=torch.float64)
            seen = 0
            for ix in order_for(data.lengths, epoch):
                total_steps += 1
                optimizer.param_groups[0]["lr"] = lr * min(1, total_steps / 80)
                p, q = data.inputs(ix, given[ix], device)
                y = data.targets(ix, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ink = net.ink(features)
                    ll, valid = net.likelihood(ink, p, q, y, net.write_key(ink[memory]), ids)
                    nll = -ll.masked_select(valid).sum()
                    n = valid.sum()
                (nll / n).backward()
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                totals += torch.stack((nll.detach(), n))
                seen += len(ix)
                if total_steps % 400 == 0:
                    print(arm, phase, total_steps, round(monotonic()-started, 1), flush=True)
                if torch.cuda.max_memory_allocated() > 12 * 1024**3 or monotonic() - started > 3600:
                    raise RuntimeError("Registered per-arm resource limit exceeded")
            loss, count = totals.tolist()
            assert seen == len(data.y) and int(count) == seen + data.lengths.sum()
            rows.append(dict(phase=phase, epoch=epoch+1, records=seen, targets=int(count),
                             mean_nll=loss/count, steps=total_steps, last_grad_norm=float(norm)))
            print(arm, rows[-1], flush=True)
    weights = directory / "final.safetensors"
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in net.state_dict().items()}, str(weights))
    write_json(directory / "fit.json", dict(epochs=rows, seconds=monotonic()-started,
               peak_bytes=torch.cuda.max_memory_allocated(), parameters=sum(p.numel() for p in net.parameters()),
               sha256=sha(weights)))
    del net, optimizer


def load_writer(path, device):
    net = GroundedGlyphDecoder().to(device).eval()
    assert sha(path / "final.safetensors") == load(path / "fit.json")["sha256"]
    net.load_state_dict(load_file(str(path / "final.safetensors")), strict=True)
    return net


@torch.inference_mode()
def evaluate(arm, bank, memory, ds, groups, device):
    net = load_writer(WORK / arm, device)
    ink = net.ink(bank.features(device))
    keys = net.write_key(ink[memory])
    lookup = torch.full((len(ink),), -1, device=device)
    lookup[memory] = torch.arange(len(memory), device=device)
    result = {}
    for mode in ("original", "rotated"):
        given = np.arange(len(ds.p))
        if mode == "rotated":
            given = np.roll(given, 1)
        cases, streams = [], []
        for j in range(0, len(given), 32):
            ix = np.arange(j, min(j+32, len(given)))
            p, q = ds.inputs(ix, given[ix], device)
            predicted, stopped = net.generate(ink, p, q, keys, memory, limit=LIMIT)
            # Targets enter only after autonomous generation. Keep unreachable NLL separate.
            y = ds.targets(ix, device)
            ll, valid = net.likelihood(ink, p, q, y, keys, lookup)
            for k, i in enumerate(ix):
                out = predicted[k].cpu().numpy()
                out = out[out != 0].tolist()
                ref = ds.y[i].tolist()
                seen = set(ds.p[i]) | set(ds.q[i])
                own = Counter(t for t in ref if t not in seen)
                produced = Counter(t for t in out if t not in seen)
                likelihood = ll[k][valid[k]]
                reachable = likelihood > -9000
                cases.append(dict(group=groups[i], exact=int(bool(stopped[k]) and out == ref),
                    stopped=int(stopped[k]), first=int(out[:1] == ref[:1]), first4=int(out[:4] == ref[:4]),
                    all_cells=len(ref), output_cells=len(out), edit=int(distance(out, ref)),
                    absent_match=sum((own & produced).values()),
                    reachable=int(reachable.sum()), targets=len(likelihood),
                    reachable_nll=float(-likelihood[reachable].sum()), first_nll=float(-likelihood[0])))
                streams.append(np.array(out, np.int32))
        np.savez_compressed(WORK / arm / (mode + ".npz"), cells=np.concatenate(streams),
                            offsets=np.r_[0, np.cumsum(list(map(len, streams)))])
        write_json(WORK / arm / (mode + ".json"), cases)
        result[mode] = {}
        for group in ("all", *sorted(set(groups))):
            records = [r for r in cases if group == "all" or r["group"] == group]
            result[mode][group] = dict(records=len(records), **{k:sum(r[k] for r in records)
                            for k in records[0] if k not in ("group", "first_nll")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.register:
        if REG.exists():
            raise FileExistsError(REG)
        write_json(REG, bindings())
        return
    if load(REG) != bindings() or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Changed sources or device")
    with (WORK.parent / "grounded_glyph_writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        WORK.mkdir()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        device = "cuda:0"
        bank, large = read_large()
        adapt, _ = small(bank, "train")
        memory = np.unique(np.concatenate([*large.y, *adapt.y])).astype(np.int64)
        assert memory[0] > 0
        np.save(WORK / "memory.npy", np.frombuffer(b"".join(bank.tiles[j] for j in memory), np.uint8).reshape(-1,128))
        mem = torch.as_tensor(memory, device=device)
        ids = torch.full((len(bank.tiles),), -1, device=device)
        ids[mem] = torch.arange(len(mem), device=device)
        features = bank.features(device)
        for arm in ("paired", "deranged"):
            fit(arm, {"large":large, "adapt":adapt}, features, mem, ids, device)
        # Development-only tile addresses can never be selected by the writing softmax.
        dev, groups = small(bank, "development")
        np.save(WORK / "evaluation_tiles.npy", np.frombuffer(b"".join(bank.tiles),np.uint8).reshape(-1,128))
        report = {arm:evaluate(arm, bank, mem, dev, groups, device) for arm in ("paired","deranged")}
        write_json(WORK / "results.json", report)
        for arm, result in report.items():
            print(arm, "original", result["original"]["all"], flush=True)


if __name__ == "__main__":
    main()
