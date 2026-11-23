"""Train two matched residual heads on fixed real pictured-question and page features."""

import json
from time import monotonic
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from safetensors.torch import load_file, save_file
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.ordered_page_features import contextualize
from ilm.visual_lm.paired_question_encoder import PicturedQuestionEncoder
from ilm.visual_lm.question_evidence_head import QuestionEvidenceHead
from scripts.mine_stopped_book_replies import BOOK
from scripts.prepare_bilingual_book_intervals import (
    WORK as INTERVAL,
    PUBLIC as INTERVAL_REPORT,
)
from scripts.reserve_mlqa_pictures import WORK, PUBLIC, CHECK
from scripts.train_bilingual_book_intervals import interval_loss
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_pictured_question_alignment import PUBLIC as QREPORT
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES, ROOT

SEED = 241206


def bodies():
    folder = INTERVAL / "aligned"
    fit = json.loads((INTERVAL_REPORT / "aligned_fit.json").read_text())
    assert all(sha(folder / k) == v for k, v in fit["weights"].items())
    reader = FontInvariantReader.restore(folder, "cuda")
    weights = ROOT / "models/pictured_question_alignment/weights.safetensors"
    assert (
        sha(weights)
        == json.loads((QREPORT / "aligned_fit.json").read_text())["weights_sha256"]
    )
    model = PicturedQuestionEncoder().cuda().eval()
    model.load_state_dict(load_file(str(weights)))
    return reader, model


@torch.no_grad()
def features():
    reader, model = bodies()
    manifest = json.loads((INTERVAL_REPORT / "preparation.json").read_text())
    assert (
        sha(INTERVAL / "training.json") == manifest["training_sha"]
        and sha(BOOK / "train.jsonl") == manifest["source_sha"]
    )
    source = {q["id"]: q for q in json.loads((INTERVAL / "training.json").read_text())}
    while any(q["swapped"] not in source for q in source.values()):
        source = {k: q for k, q in source.items() if q["swapped"] in source}
    face = Face(FACES[0])
    pairs = []
    indices = {}
    true = []
    swaps = []
    targets = []
    for line in (BOOK / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        p = face(row["page"])
        qs = {q["id"]: q for q in row["questions"]}
        for k, q in qs.items():
            if k not in source:
                continue
            for lang in ("zh", "en"):
                genuine = face(q[lang])
                other = face(qs[source[k]["swapped"]][lang])
                pairs.append((p, genuine))
                true.append(indices.setdefault(genuine, len(indices)))
                swaps.append(indices.setdefault(other, len(indices)))
                targets.append(source[k]["intervals"])
    assert len(pairs) == 22876 and sorted(true) == sorted(swaps)
    with torch.inference_mode():
        vectors = model.encode(list(indices)).cpu()
    states = []
    for i in range(0, len(pairs), 64):
        states.extend(
            state[:, :770].contiguous()
            for _, state in contextualize(reader, pairs[i : i + 64])
        )
    return states, vectors.clone(), np.asarray(true), np.asarray(swaps), targets


def fit(arm, data):
    states, vectors, true, swaps, targets = data
    path = WORK / (arm + ".safetensors")
    if path.exists():
        raise FileExistsError(path)
    which = true if arm == "matched" else swaps
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    head = QuestionEvidenceHead().cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, fused=True)
    start = monotonic()
    torch.cuda.reset_peak_memory_stats()
    history = []
    updates = 0
    for _ in range(4):
        order = rng.permutation(len(states))
        total = 0.0
        for i in range(0, len(order), 48):
            ids = order[i : i + 48]
            selected = [states[k] for k in ids]
            h = pad_sequence(selected, batch_first=True).cuda()
            mask = (
                torch.arange(h.shape[1], device="cuda")[None, :]
                < torch.tensor(list(map(len, selected)), device="cuda")[:, None]
            )
            q = vectors[which[ids]].cuda()
            score = head(h, q)
            loss = interval_loss(*score.unbind(-1), mask, [targets[k] for k in ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            updates += 1
            total += loss.item() * len(ids)
            if (
                monotonic() - start > 1200
                or torch.cuda.max_memory_allocated() > 12 * 2**30
            ):
                raise RuntimeError("Evidence head exceeded fit allocation")
        history.append(total / len(states))
        print(arm, len(history), history[-1], flush=True)
    tmp = path.with_suffix(".partial")
    save_file({k: v.detach().cpu() for k, v in head.state_dict().items()}, str(tmp))
    tmp.rename(path)
    record(
        PUBLIC / (arm + "_fit.json"),
        dict(
            arm=arm,
            trainable=sum(p.numel() for p in head.parameters()),
            updates=updates,
            seconds=monotonic() - start,
            peak_bytes=torch.cuda.max_memory_allocated(),
            epochs=4,
            n=len(states),
            mean_nll=history,
            sha256=sha(path),
            seed=SEED,
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        assert CHECK.is_file()
        start = monotonic()
        data = features()
        record(
            PUBLIC / "features.json",
            dict(
                seconds=monotonic() - start,
                storage_bytes=sum(x.nelement() * x.element_size() for x in data[0]),
                reservation=sha(CHECK),
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            ),
        )
        for arm in ("matched", "exchanged"):
            fit(arm, data)
