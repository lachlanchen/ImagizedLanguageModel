"""Fit the current-field glyph realizer without inherited language weights."""

import hashlib
from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.field_cursor import FieldCursor
from scripts.prepare_chinese_two_fact import (
    JOINED_SCHEDULE,
    WORK,
    MODEL,
    PARENT,
    PUB,
    FIELDS,
    MARKS,
    read,
    sha,
    dump,
)
from scripts.train_chinese_two_fact import (
    gpu_owner,
    snapshot,
    consumed,
    archive,
    output_bank,
)


def layout():
    chars = read(MODEL / "alphabet.json")
    return [c for c in chars if c != MARKS[2]]


def labels(device):
    packed = np.load(PARENT / "label-pixels.npy")
    return torch.as_tensor(
        observations({"tiles": packed.reshape(-1, 128)}, "ink"), device=device
    ).reshape(len(packed), -1, 256)


def palette(device):
    chars = read(MODEL / "alphabet.json")
    return output_bank(device)[[i for i, c in enumerate(chars) if c != MARKS[2]]]


def cursor_actions(sequences, original):
    chars = [c for c in original if c != MARKS[2]]
    ids = {c: i for i, c in enumerate(original)}
    result = []
    for seq in sequences:
        count = 0
        steps = []
        for idx in seq:
            char = chars[idx]
            if char == MARKS[1]:
                if count > 1:
                    raise ValueError("Too many copies in the field program")
                char = MARKS[1 + count]
                count += 1
            steps.append(ids[char])
        result.append(steps)
    return result


def convert_plans(plans, total):
    # The corpus reserves len(FIELDS) for no second field. The realizer
    # embeds domain labels in the same bank, so its absent-field index is total.
    return plans.masked_fill(plans == len(FIELDS), total)


def load_cursor(device):
    p = MODEL / "cursor.safetensors"
    receipt = read(PUB / "cursor-fit.json")
    if sha(p) != receipt["weights"]:
        raise ValueError("Changed cursor weights")
    chars = layout()
    net = FieldCursor(len(chars), chars.index(MARKS[1])).to(device)
    net.load_state_dict(load_file(str(p), device=device))
    return net.eval()


def main():
    meta = read(PUB / "corpus.json")
    if (PUB / "cursor-fit.json").exists():
        load_cursor("cpu")
        return
    rng = np.random.default_rng(6745)
    torch.manual_seed(58043)
    old = read(MODEL / "alphabet.json")
    chars = layout()
    convert = np.array(
        [chars.index(c if c != MARKS[2] else MARKS[1]) for c in old], np.int16
    )
    sets = []
    with gpu_owner():
        field_labels = labels("cuda")
        p = palette("cuda")
        nfields = len(FIELDS)
        net = FieldCursor(len(chars), chars.index(MARKS[1])).cuda()
        opt = torch.optim.AdamW(
            net.parameters(), lr=0.0025, fused=True, weight_decay=0.01
        )
        for split in ("train", "joined"):
            d = archive(WORK / (split + ".npz"), meta["archives"][split])
            y = d["answer"]
            y = np.where(y < 0, -1, convert[y.clip(min=0)])
            sets.append(
                tuple(
                    torch.as_tensor(x, device="cuda").long()
                    for x in (y, d["domain"] + nfields, d["fields"])
                )
            )
        n, m = meta["splits"]["train"]
        pools = (np.arange(n), np.arange(n, n + m), np.arange(meta["joined"]))
        start = monotonic()
        energy = snapshot()
        torch.cuda.reset_peak_memory_stats()
        digest = hashlib.sha256()
        positions = 0
        for t in range(2400):
            group = (0, 1, 0, 2)[t % 4]
            y, domain, fields = sets[int(group == 2)]
            ix = rng.choice(pools[group], size=48)
            digest.update(bytes([group]))
            digest.update(ix.tobytes())
            target = y[ix]
            sizes = (target >= 0).sum(1)
            target = target[:, : sizes.max().item()]
            previous = F.pad(target[:, :-1].clamp_min(0), (1, 0))
            plans = convert_plans(fields[ix], len(field_labels))
            logits = net(
                net.encode_labels(field_labels),
                domain[ix],
                plans,
                previous,
                net.glyph(p),
            )
            loss = F.cross_entropy(
                logits.flatten(0, 1), target.flatten(), ignore_index=-1
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if t == 1600:
                opt.param_groups[0]["lr"] = 0.0007
            if t % 300 == 0 and (
                not torch.isfinite(loss)
                or monotonic() - start > 300
                or torch.cuda.max_memory_allocated() > 2**30
            ):
                raise RuntimeError("Cursor fit exceeded resource limit")
            positions += sizes.sum().item()
        torch.cuda.synchronize()
        seconds = monotonic() - start
        cost = consumed(energy, snapshot())
        if digest.hexdigest() != JOINED_SCHEDULE:
            raise ValueError("Training order differs")
        path = MODEL / "cursor.safetensors"
        tmp = path.with_suffix(".tmp")
        save_file({k: v.cpu() for k, v in net.state_dict().items()}, str(tmp))
        tmp.replace(path)
        dump(
            PUB / "cursor-fit.json",
            dict(
                weights=sha(path),
                parameters=sum(p.numel() for p in net.parameters()),
                updates=2400,
                positions=positions,
                seconds=seconds,
                energy=cost,
                schedule=digest.hexdigest(),
                peak=torch.cuda.max_memory_allocated(),
            ),
        )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
