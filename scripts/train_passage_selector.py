"""Fit the two registered page-to-word models on a single shared GPU."""

from hashlib import sha256
from pathlib import Path
from time import monotonic
import json

import numpy as np
import torch
from safetensors.torch import save_file

from ilm.visual_lm.passage_selector import PassageSelector, Print, VisualBank
from scripts.prepare_passage_selector import MODEL, PUB, ROOT, RULE, SEED, WORK, base
from scripts.train_open_news_writer import atomic_save, gpu_owner, configure_cuda_cli

RATES = (0.0008, 0.0005, 0.0003, 0.00015)


def bindings():
    files = (
        RULE,
        PUB / "data.json",
        MODEL / "document_frequency.json",
        Path(__file__),
        ROOT / "scripts/prepare_passage_selector.py",
        ROOT / "ilm/visual_lm/passage_selector.py",
        ROOT / "ilm/visual_lm/ordered_translation.py",
        ROOT / "ilm/visual_lm/portable_writer.py",
    )
    return {str(p.relative_to(ROOT)): base.sha(p) for p in files}


def load_bank(device):
    face = Print()
    words = [face.ids(w) for w in json.loads((WORK / "forms.json").read_text())]
    return VisualBank(face, words, device)


def shuffle_distant(src, marks, generator):
    pos = torch.arange(src.shape[1], device=src.device)
    key = torch.rand(src.shape, device=src.device, generator=generator) + 100
    key = torch.where(src == 0, pos - 200, key)
    key = torch.where(pos >= len(pos) - 6, pos + 300, key)
    order = key.argsort(1)
    return src.gather(1, order), marks.gather(1, order)


def fit(arm, bank, data):
    out = MODEL / arm
    if (out / "fit.json").exists():
        existing = base.read(out / "fit.json")
        if existing["bindings"] != bindings() or existing["weights"] != base.sha(
            out / "weights.safetensors"
        ):
            raise ValueError(f"Changed completed fit: {arm}")
        return existing
    torch.manual_seed(SEED)
    net = PassageSelector(bank.face.coarse)
    initial = sha256()
    for p in net.parameters():
        initial.update(p.detach().numpy().tobytes())
    net.cuda().train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=RATES[0], fused=True)
    rng = np.random.default_rng(SEED)
    shuffle_rng = torch.Generator(device="cuda").manual_seed(SEED + 1)
    start = monotonic()
    torch.cuda.reset_peak_memory_stats()
    history = []
    for epoch, rate in enumerate(RATES, 1):
        order = rng.permutation(len(data["source"])).astype("<i4")
        indices = torch.as_tensor(order, device="cuda").long()
        optimizer.param_groups[0]["lr"] = rate
        metrics = torch.zeros(4, device="cuda", dtype=torch.float64)
        for step, ix in enumerate(indices.split(128), 1):
            src = data["source"][ix].long()
            marks = data["marks"][ix].long()
            if arm == "shuffled":
                src, marks = shuffle_distant(src, marks, shuffle_rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, logmass, present = net.training_loss(
                    bank,
                    src,
                    marks,
                    data["task"][ix].long(),
                    data["target"][ix],
                    data["end"][ix],
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            metrics += torch.stack(
                (
                    loss.detach(),
                    -logmass[present].sum(),
                    -logmass[~present].sum(),
                    present.sum(),
                )
            )
            if (
                monotonic() - start > 2700
                or torch.cuda.max_memory_allocated() > 16 * 2**30
            ):
                raise RuntimeError(
                    f"Registered fit budget exceeded at {arm}/{epoch}/{step}"
                )
            if step % 480 == 0:
                print(arm, epoch, step, round(monotonic() - start, 1), flush=True)
        objective, in_page, out_page, n = metrics.tolist()
        history.append(
            dict(
                epoch=epoch,
                rows=len(order),
                steps=step,
                rate=rate,
                objective=objective / step,
                present=int(n),
                absent=len(order) - int(n),
                present_nll=in_page / n,
                absent_nll=out_page / (len(order) - n),
                schedule=sha256(order.tobytes()).hexdigest(),
                elapsed=monotonic() - start,
            )
        )
        print(history[-1], flush=True)
    out.mkdir(exist_ok=True)
    weights = out / "weights.safetensors"
    atomic_save(
        lambda f: save_file(
            {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, f
        ),
        weights,
    )
    receipt = dict(
        arm=arm,
        initial=initial.hexdigest(),
        epochs=history,
        bindings=bindings(),
        weights=base.sha(weights),
        device=torch.cuda.get_device_name(),
        peak_bytes=torch.cuda.max_memory_allocated(),
        seconds=monotonic() - start,
        parameters=sum(p.numel() for p in net.parameters()),
    )
    base.dump(out / "fit.json", receipt)
    return receipt


def main():
    configure_cuda_cli()
    receipt = base.read(PUB / "data.json")
    assert receipt["script"] == base.sha(ROOT / "scripts/prepare_passage_selector.py")
    assert receipt["protocol"] == base.sha(RULE) and all(
        base.sha(WORK / k) == v for k, v in receipt["files"].items()
    )
    assert receipt["prior_df"] == base.sha(MODEL / "document_frequency.json")
    with gpu_owner():
        bank = load_bank("cuda")
        with np.load(WORK / "train.npz") as archive:
            data = {
                k: torch.as_tensor(archive[k], device="cuda")
                for k in ("source", "marks", "task", "end")
            }
            data["target"] = torch.as_tensor(
                archive["target"][:, 0], device="cuda"
            ).long()
        assert bank.lengths[data["target"]].min() > 0
        a, b = [fit(arm, bank, data) for arm in ("intact", "shuffled")]
        assert a["initial"] == b["initial"] and all(
            x["schedule"] == y["schedule"]
            for x, y in zip(a["epochs"], b["epochs"], strict=True)
        )


if __name__ == "__main__":
    main()
