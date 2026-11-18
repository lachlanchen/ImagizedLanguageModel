"""Hold target order and optimizer work fixed and replace new questions by original questions."""

from collections import defaultdict
import fcntl
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ilm.visual_lm.conditioned_entry_reader import ConditionedEntryReader
from ilm.visual_lm.entry_span_writer import parse_card
from scripts.check_entry_question_order import produce
from scripts.evaluate_independent_entry_reader import views
from scripts.evaluate_paired_student import save_streams
from scripts.train_independent_entry_reader import (
    MODELS,
    OUT,
    PARENT,
    ROOT,
    SEED,
    BATCH,
    EPOCHS,
    WordingRequests,
    binding,
    load,
    sha,
    write_json,
    corpus,
    assets,
)


def control():
    assert load(OUT / "registration.json")["sources"] == binding()
    config = load(OUT / "replay_protocol.json")
    assert config["script_sha"] == sha(Path(__file__))
    weight = MODELS / "replay.safetensors"
    if weight.exists():
        raise FileExistsError(weight)
    data = WordingRequests()
    n = len(data.target) // 2
    originals = defaultdict(list)
    for i, t in enumerate(data.target[:n]):
        originals[t].append(i)
    count = defaultdict(int)
    remap = []
    for t in data.target[n:]:
        remap.append(originals[t][count[t]])
        count[t] += 1
    ids = np.asarray(remap)
    assert sorted(ids) == list(range(n)) and np.array_equal(
        data.target[ids], data.target[n:]
    )
    for tensor in (data.strips["ordered"], data.anchors):
        tensor[n:] = tensor[data.tensor(ids)]
    torch.manual_seed(SEED)
    m = ConditionedEntryReader(False).cuda()
    missing, extra = m.load_state_dict(
        load_file(str(PARENT / "ordered.safetensors")), strict=False
    )
    assert missing == ["gain"] and not extra
    opt = torch.optim.AdamW(m.parameters(), lr=0.0008)
    rng = np.random.default_rng(SEED)
    losses = []
    updates = 0
    torch.cuda.reset_peak_memory_stats()
    start = monotonic()
    for _ in range(EPOCHS):
        order = rng.permutation(len(data.target))
        numer, denom = 0.0, 0
        for offset in range(0, len(order), BATCH):
            idx = data.tensor(order[offset : offset + BATCH])
            y = data.answer_id[idx]
            keys = data.keys(m, "ordered", idx)
            scores, _ = m(keys, data.history[y])
            allowed = data.gold[y]
            active = allowed.any(-1)
            scores, allowed = scores[active], allowed[active]
            loss = (
                scores.logsumexp(-1)
                - scores.masked_fill(~allowed, -torch.inf).logsumexp(-1)
            ).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            numer += loss.item() * len(scores)
            denom += len(scores)
            updates += 1
        losses.append(numer / denom)
        assert (
            monotonic() - start < 1200 and torch.cuda.max_memory_allocated() < 8 * 2**30
        )
        print(len(losses), losses[-1], flush=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()},
        str(weight),
    )
    write_json(
        OUT / "replay_fit.json",
        dict(
            weight_sha=sha(weight),
            loss=losses,
            updates=updates,
            exposures=EPOCHS * len(data.target),
            seconds=monotonic() - start,
            peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            target_matched_rows=len(ids),
            distinct_original_questions_duplicated=n,
        ),
    )
    del opt, data
    m = m.cpu().eval()
    src = corpus("development")
    marks, lit = assets()
    records = [parse_card(c, marks) for c in src["cards"]]
    result = {}
    for name, qs, rows, group in views():
        pred, done = produce(m, qs, records, rows // 8, lit)
        good = np.asarray(
            [
                s and x == src["answers"][r]
                for x, s, r in zip(pred, done, rows, strict=True)
            ]
        )
        save_streams(
            name + "_replay",
            [np.frombuffer(x, np.uint8).reshape(-1, 128) for x in pred],
            OUT / "outputs",
        )
        result[name] = dict(
            exact=int(good.sum()),
            complete=int(good.reshape(-1, group).all(1).sum()),
            n=len(good),
            rows=np.flatnonzero(good).tolist(),
            stops=done,
            stream_sha=sha(OUT / "outputs" / (name + "_replay.npz")),
        )
        print(name, result[name]["exact"], flush=True)
    write_json(OUT / "replay_evaluation.json", result)


if __name__ == "__main__":
    torch.set_num_threads(2)
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control()
