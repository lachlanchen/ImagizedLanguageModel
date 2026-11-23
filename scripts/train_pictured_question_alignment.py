"""Learn bilingual compatibility from the images of questions alone."""

import json
from time import monotonic
import numpy as np
from safetensors.torch import save_file
import torch
from ilm.visual_lm.paired_question_encoder import PicturedQuestionEncoder
from scripts.mine_stopped_book_replies import BOOK
from scripts.prepare_bilingual_book_intervals import WORK as SOURCE, PUBLIC as INTERVALS
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES, ROOT

WORK = ROOT / "artifacts/pictured_question_alignment_v1"
PUBLIC = ROOT / "references/evidence/pictured_question_alignment_20261124"
SEED = 241193


def data():
    binding = json.loads((INTERVALS / "preparation.json").read_text())
    assert (
        sha(SOURCE / "training.json") == binding["training_sha"]
        and sha(BOOK / "train.jsonl") == binding["source_sha"]
    )
    swapped = {
        x["id"]: x["swapped"]
        for x in json.loads((SOURCE / "training.json").read_text())
    }
    while any(v not in swapped for v in swapped.values()):
        swapped = {k: v for k, v in swapped.items() if v in swapped}
    face = Face(FACES[0])
    streams = {}
    pages, rows = [], []
    for line in (BOOK / "train.jsonl").read_text().splitlines():
        entry = json.loads(line)
        qs = {q["id"]: q for q in entry["questions"]}
        indices = []
        for key, q in qs.items():
            if key in swapped:
                texts = (q["zh"], q["en"], qs[swapped[key]]["en"])
                values = []
                for t in texts:
                    raw = face(t)
                    values.append(streams.setdefault(raw, len(streams)))
                indices.append(len(rows))
                rows.append(values)
        if indices:
            pages.append(indices)
    assert len(rows) == 11438
    return pages, np.asarray(rows), PicturedQuestionEncoder.pack(streams, "cuda")


def loss_for(logits, z, e):
    same = (z[:, None] == z[None, :]) | (e[:, None] == e[None, :])
    gold = logits.masked_fill(~same, -torch.inf)
    return sum((logits.logsumexp(i) - gold.logsumexp(i)).mean() for i in (0, 1)) / 2


def fit(arm, pages, rows, packed):
    dest = WORK / (arm + ".safetensors")
    if dest.exists():
        raise FileExistsError(dest)
    ids, pixels = packed
    torch.manual_seed(SEED)
    model = PicturedQuestionEncoder().cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0005, weight_decay=0.01, fused=True
    )
    rng = np.random.default_rng(SEED)
    start = monotonic()
    torch.cuda.reset_peak_memory_stats()
    updates = 0
    history = []
    for epoch in range(8):
        batches = [[]]
        for i in rng.permutation(len(pages)):
            if len(batches[-1]) + len(pages[i]) > 96:
                batches.append([])
            batches[-1].extend(pages[i])
        total = 0.0
        for batch in batches:
            z = rows[batch, 0]
            e = rows[batch, 1 if arm == "aligned" else 2]
            queries = [ids[i] for i in np.concatenate((z, e))]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, b = model(queries, pixels).chunk(2)
            objective = loss_for(
                a @ b.T / 0.07, torch.from_numpy(z).cuda(), torch.from_numpy(e).cuda()
            )
            if not torch.isfinite(objective):
                raise ValueError("Invalid alignment loss")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total += objective.item() * len(batch)
            updates += 1
            if (
                monotonic() - start > 1200
                or torch.cuda.max_memory_allocated() > 12 * 2**30
            ):
                raise RuntimeError("Question fit exceeded allocation")
        history.append(total / len(rows))
        print(arm, epoch + 1, history[-1], flush=True)
    temporary = dest.with_suffix(".partial")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(temporary),
    )
    temporary.rename(dest)
    record(
        PUBLIC / (arm + "_fit.json"),
        dict(
            rows=len(rows),
            pages=len(pages),
            updates=updates,
            epochs=8,
            params=sum(p.numel() for p in model.parameters()),
            seconds=monotonic() - start,
            loss=history,
            peak_bytes=torch.cuda.max_memory_allocated(),
            weights_sha256=sha(dest),
            seed=SEED,
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        WORK.mkdir(exist_ok=True)
        PUBLIC.mkdir(exist_ok=True)
        sources = data()
        for arm in ("aligned", "exchanged"):
            fit(arm, *sources)
