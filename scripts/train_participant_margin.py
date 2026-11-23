"""Fit the role-conditioned span scorer and the matched assignment control on one GPU."""

import argparse
import fcntl
import json
import os
from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from ilm.visual_lm.bound_fact_reader import BoundFactReader, features_batch
from ilm.visual_lm.ordered_page_features import contextualize
from ilm.visual_lm.role_conditioned_span import ParticipantHead, span_scores
from scripts.prepare_participant_margin import OUT, PUBLIC, SEED
from scripts.prepare_visual_facts import Face, FONTS, ROOT, RELATIONS, WORK, write_json
from scripts.train_visual_facts import rows, READER, visual_cache
from scripts.train_pointing_causal_writer import sha


def read(name):
    path = OUT / (name + ".json")
    expected = json.loads((PUBLIC / "corpus.json").read_text())["sha256"][name]
    if sha(path) != expected:
        raise ValueError("Split differs from the corpus receipt")
    return json.loads(path.read_text())


def picture_cache(reader, face, items):
    result = []
    for i in range(0, len(items), 128):
        chunk = items[i : i + 128]
        result.extend(
            contextualize(
                reader, [(face(x["p"]), face(x["q"])) for x in chunk], batch_size=24
            )
        )
    return result


def supervision(data, control):
    rng = np.random.default_rng(SEED + 4)
    labels = []
    for x in data:
        p = x["p"]
        correct = x["o"]
        rival = x.get("rival")
        if control and rival and rng.random() < 0.5:
            correct, rival = rival, correct
        a = p.index(correct)
        b = p.index(rival) if rival else a
        labels.append(
            (
                RELATIONS.index(x["r"]),
                a,
                len(correct) - 1,
                b,
                len(rival or correct) - 1,
                rival is not None,
            )
        )
    return torch.tensor(labels, device="cuda")


def fit(mode, reader, data, features):
    torch.manual_seed(SEED)
    head = ParticipantHead(len(RELATIONS)).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=0.0007, weight_decay=0.01)
    y = supervision(data, mode == "permuted")
    rng = np.random.default_rng(SEED + 5)
    t = monotonic()
    torch.cuda.reset_peak_memory_stats()
    history = []
    steps = 0
    for epoch in range(5):
        loss_sum = 0.0
        for pool in np.array_split(
            rng.permutation(len(data)), max(1, len(data) // 768)
        ):
            order = sorted(pool, key=lambda i: len(features[i][1]))
            for k in range(0, len(order), 48):
                ids = order[k : k + 48]
                cls, page, valid = features_batch([features[i] for i in ids], "cuda")
                labels = y[ids]
                with torch.no_grad():
                    _, old = reader.head(cls, page, valid)
                logits = head(page, valid, labels[:, 0], old[:, :, 2:])
                score = span_scores(logits, valid)
                w = score.shape[2]
                target = labels[:, 1] * w + labels[:, 2]
                flat = score.flatten(1)
                idx = torch.arange(len(ids), device=y.device)
                nll = F.cross_entropy(flat, target)
                margin = F.relu(
                    1 + flat[idx, labels[:, 3] * w + labels[:, 4]] - flat[idx, target]
                )
                flag = labels[:, -1]
                loss = nll
                if mode != "likelihood":
                    loss = loss + (margin * flag).sum() / flag.sum().clamp_min(1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
                opt.step()
                loss_sum += float(loss.detach()) * len(ids)
                steps += 1
        history.append(loss_sum / len(data))
        print(mode, epoch + 1, history[-1], flush=True)
    path = OUT / (mode + ".safetensors")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()},
        str(path),
    )
    write_json(
        PUBLIC / (mode + "_fit.json"),
        dict(
            weights=sha(path),
            records=len(data),
            updates=steps,
            loss=history,
            seconds=monotonic() - t,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            trainables=sum(p.numel() for p in head.parameters()),
        ),
    )


def main(ablation=False):
    arms = ("likelihood",) if ablation else ("matched", "permuted")
    if any((OUT / (s + ".safetensors")).exists() for s in arms):
        raise FileExistsError("Existing fit")
    base = BoundFactReader(READER, WORK / "genuine", "cuda")
    face = Face((FONTS[0], 2))
    old = rows("training")
    replay = read("replay")
    paired = read("paired")
    oldfeatures = visual_cache("training", old, face, base.reader)
    values = [oldfeatures[i] for i in replay]
    del oldfeatures
    start = monotonic()
    values.extend(picture_cache(base.reader, face, paired))
    write_json(
        PUBLIC / ("likelihood_features.json" if ablation else "features.json"),
        dict(
            new_pairs=len(paired),
            reused=len(replay),
            elapsed_seconds=monotonic() - start,
        ),
    )
    data = [old[i] for i in replay] + paired
    for arm in arms:
        fit(arm, base, data, values)


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU0 required")
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(3)
        parser = argparse.ArgumentParser()
        parser.add_argument("--likelihood-only", action="store_true")
        main(parser.parse_args().likelihood_only)
