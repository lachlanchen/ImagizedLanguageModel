"""Train open glyph generation and the matched two-page reference preference trial."""

from collections import defaultdict
import hashlib
from pathlib import Path
import math
from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from ilm.visual_lm.discourse_writer import DiscourseWriter
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_visual_discourse import (
    BASE,
    PUBLIC,
    MODEL,
    WINDOW,
    read,
    dump,
    sha,
)
from scripts.measure_writer_energy import snapshot, consumed
from scripts.train_span_choice import gpu_owner

WARM = 8000
TUNE = 2000
SEED = 918032


class Corpus:
    def __init__(self):
        self.meta = read(PUBLIC / "corpus.json")
        for name, tag in (("corpus.npz", "corpus"), ("records.json", "records")):
            if sha(BASE / name) != self.meta[tag]:
                raise ValueError("Changed corpus: " + name)
        with np.load(BASE / "corpus.npz") as z:
            self.stream = z["stream"]
            self.tiles = z["tiles"]
            self.chars = z["characters"]
        self.n = self.meta["train_shapes"] + 2
        self.rows = read(BASE / "records.json")
        groups = defaultdict(list)
        pairs = defaultdict(list)
        for i, r in enumerate(self.rows):
            if r["split"] == "train":
                groups[r["task"]].append(i)
                if "pair" in r:
                    pairs[r["pair"]].append(i)
        self.groups = {k: np.array(v) for k, v in groups.items()}
        self.pairs = np.array(list(pairs.values()))
        self.books = {}
        for k in ("book-en", "book-zh"):
            weight = np.array(
                [self.rows[i]["end"] - self.rows[i]["start"] for i in self.groups[k]],
                float,
            )
            self.books[k] = weight / weight.sum()

    def feature(self, mode, device):
        by = np.zeros((len(self.chars), 4), np.uint8)
        if mode == "byte":
            for i, ch in enumerate(self.chars):
                v = ch.encode()
                by[i, : len(v)] = list(v)
        x = observations(dict(tiles=self.tiles, bytes=by), mode)
        return torch.as_tensor(x, device=device)

    def seq(self, i):
        r = self.rows[i]
        return self.stream[r["start"] : r["end"]]

    def ordinary(self, rng, counts):
        sequences = []
        masks = []
        for k, count in counts.items():
            for i in rng.choice(self.groups[k], count, p=self.books.get(k)):
                r = self.rows[i]
                ids = self.seq(i)
                start = 0
                if len(ids) > WINDOW:
                    if k.startswith("instruction") and rng.random() < 0.45:
                        start = min(len(ids) - WINDOW, max(0, r["cut"] - WINDOW // 2))
                    else:
                        start = int(rng.integers(0, len(ids) - WINDOW + 1))
                data = ids[start : start + WINDOW]
                cut = r["cut"] - start if k == "reference" else 1
                sequences.append(data)
                masks.append(np.arange(len(data)) >= cut)
        return sequences, masks

    def batch(self, step, fine):
        rng = np.random.default_rng([SEED + int(fine), step])
        counts = {
            "book-en": 5,
            "book-zh": 5,
            "instruction-en": 8,
            "instruction-zh": 8,
            "reference": 6,
        }
        if fine:
            counts = {
                "book-en": 2,
                "book-zh": 2,
                "instruction-en": 5,
                "instruction-zh": 5,
                "reference": 2,
            }
        seq, masks = self.ordinary(rng, counts)
        if fine:
            foils = []
            span = []
            for i in self.pairs[rng.integers(0, len(self.pairs), 8)].ravel():
                r = self.rows[i]
                s = self.seq(i)
                cut = r["cut"]
                seq.append(s)
                masks.append(np.arange(len(s)) >= cut)
                alt = np.r_[s[:cut], np.asarray(r["foil"], np.uint16)]
                foils.append(alt)
                span.append(np.arange(len(alt)) >= cut)
            seq.extend(foils)
            masks.extend(span)
        ids = np.zeros((len(seq), WINDOW), np.int32)
        active = np.zeros_like(ids, dtype=bool)
        for j, (s, m) in enumerate(zip(seq, masks, strict=True)):
            ids[j, : len(s)] = s
            active[j, : len(s)] = m
        return ids, active


def load_model(arm, device):
    weights = MODEL / (arm + ".safetensors")
    if sha(weights) != read(PUBLIC / (arm + ".json"))["weights"]:
        raise ValueError("Changed model weights")
    model = DiscourseWriter().to(device)
    model.load_state_dict(load_file(str(weights), device=str(device)))
    return model.eval()


def loss_for(model, features, ids, mask, n, fine, paired):
    bank = model.encode(features)
    if fine:
        # Sparse answer positions on paired pages avoid materializing full-page
        # vocabulary logits for prompts whose text is not scored.
        body = model.log_prob(bank, ids[:16], n)
        ordinary = -(body * mask[:16]).sum(1) / mask[:16].sum(1).clamp_min(1)
        rest = mask[16:]
        width = int(rest.sum(1).max())
        pos = (
            rest.int().argmax(1)[:, None] + torch.arange(width, device=ids.device)
        ).clamp_max(WINDOW - 1)
        lp = model.log_prob(bank, ids[16:], n, pos)
        length = rest.sum(1)
        total = (
            lp * (torch.arange(width, device=ids.device)[None, :] < length[:, None])
        ).sum(1)
        correct = -total[:16] / length[:16]
        delta = (total[:16] - total[16:]) / torch.maximum(length[:16], length[16:])
        weaker = delta.reshape(-1, 2).amin(1)
        base = torch.cat((ordinary, correct)).mean()
        return base + (0.65 if paired else 0.0) * F.softplus(-weaker).mean()
    lp = model.log_prob(bank, ids, n)
    return -(lp * mask).sum() / mask.sum()


def fit(data, arm, mode, parent=None):
    receipt = PUBLIC / (arm + ".json")
    path = MODEL / (arm + ".safetensors")
    if receipt.exists():
        load_model(arm, "cpu")
        return
    fine = parent is not None
    steps = TUNE if fine else WARM
    torch.manual_seed(SEED)
    model = (load_model(parent, "cuda") if fine else DiscourseWriter().cuda()).train()
    features = data.feature(mode, "cuda")[: data.n]
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.03, fused=True)
    state = BASE / (arm + "-resume.pt")
    binding = (
        arm,
        sha(PUBLIC / "corpus.json"),
        sha(MODEL / (parent + ".safetensors")) if fine else None,
        sha(Path(__file__)),
        sha(Path(ROOT_MODEL)),
    )
    trace = []
    completed = 0
    elapsed = 0.0
    energy = 0.0
    digest = bytes(32)
    if state.exists():
        saved = torch.load(state, map_location="cuda", weights_only=False)
        if saved["binding"] != binding:
            raise ValueError("Resume binding differs")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        trace = saved["trace"]
        completed = saved["step"]
        elapsed = saved["elapsed"]
        energy = saved["energy"]
        digest = saved["schedule"]
        del saved
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = monotonic()
    meter = snapshot()
    for step in range(completed + 1, steps + 1):
        rate = 0.00035 if fine else 0.0009
        opt.param_groups[0]["lr"] = (
            rate
            * min(1, step / 100)
            * (0.12 + 0.88 * (1 + math.cos(math.pi * step / steps)) / 2)
        )
        ids, mask = data.batch(step, fine)
        digest = hashlib.sha256(digest + ids.tobytes() + mask.tobytes()).digest()
        ids = torch.from_numpy(ids).cuda()
        mask = torch.from_numpy(mask).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_for(
                model, features, ids, mask, data.n, fine, arm.endswith("-pair")
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 100 == 0:
            val = float(loss.detach())
            if not math.isfinite(val) or torch.cuda.max_memory_allocated() > 20 * 2**30:
                raise RuntimeError("Invalid loss or memory ceiling exceeded")
            trace.append([step, val])
            print(
                arm,
                step,
                round(val, 4),
                round(elapsed + monotonic() - start, 1),
                flush=True,
            )
        if step % 500 == 0:
            torch.cuda.synchronize()
            used = consumed(meter, snapshot())["joules"]
            if used is None:
                raise RuntimeError("Energy counter unavailable")
            elapsed += monotonic() - start
            energy += used
            tmp = state.with_suffix(".tmp")
            torch.save(
                dict(
                    binding=binding,
                    step=step,
                    model=model.state_dict(),
                    optimizer=opt.state_dict(),
                    schedule=digest,
                    trace=trace,
                    elapsed=elapsed,
                    energy=energy,
                ),
                tmp,
            )
            tmp.replace(state)
            start = monotonic()
            meter = snapshot()
    tmp = path.with_suffix(".tmp")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(tmp),
    )
    tmp.replace(path)
    dump(
        receipt,
        dict(
            weights=sha(path),
            mode=mode,
            parent=parent,
            steps=steps,
            parameters=sum(v.numel() for v in model.parameters()),
            seconds=elapsed,
            device_joules=energy,
            peak_allocated=torch.cuda.max_memory_allocated(),
            schedule=digest.hex(),
            trace=trace,
        ),
    )
    state.unlink(missing_ok=True)


from ilm.visual_lm import discourse_writer

ROOT_MODEL = discourse_writer.__file__


def main():
    data = Corpus()
    with gpu_owner():
        for mode in ("ink", "byte"):
            fit(data, mode + "-warm", mode)
            for kind in ("independent", "pair") if mode == "ink" else ("pair",):
                fit(data, mode + "-" + kind, mode, mode + "-warm")


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
