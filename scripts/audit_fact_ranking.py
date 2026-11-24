"""Check real candidate gradients against scalar enumeration and reproduce released PNGs."""

from argparse import Namespace
import numpy as np
import torch

from ilm.visual_lm.fact_ranking import candidate_loss
from scripts.prepare_fact_channel import MODEL as BASE, read, dump, sha
from scripts.prepare_fact_ranking_confirmation import WORK, PUB
from scripts.train_fact_ranking import MODEL, MODES, restore
from scripts.write_fact_channel import main as write


def main():
    torch.set_num_threads(3)
    bank = torch.load(WORK / "train-candidates.pt", weights_only=True, mmap=True)
    supported = bank["correct"].sum(1)
    ids = [int(torch.where(supported == n)[0][0]) for n in (0, 1, 2)]
    checked = {}
    for mode in MODES:
        head = restore(mode, "cpu")
        y = bank["correct"][ids]
        args = [bank[k][ids] for k in ("features", "base", "lengths")]
        z = head(*args)
        loss = candidate_loss(z, y)
        # Enumerate positive and legal positions independently; include no missing row.
        terms = []
        for row, mask in zip(z, y):
            if mask.any():
                terms.append(
                    torch.stack([val for val in row if val.isfinite()]).logsumexp(0)
                    - torch.stack(
                        [val for val, match in zip(row, mask) if match]
                    ).logsumexp(0)
                )
        other = torch.stack(terms).mean()
        params = tuple(head.parameters())
        gradients = torch.autograd.grad(loss, params, retain_graph=True)
        reference = torch.autograd.grad(other, params)
        errors = []
        for a, b in zip(gradients, reference, strict=True):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=3e-5)
            errors.append(float((a - b).abs().max()))
        checked[mode] = dict(
            rows=ids,
            matches=supported[ids].tolist(),
            loss=float(loss.detach()),
            enumerated=float(other.detach()),
            tensors=len(params),
            max_abs=max(errors),
        )
    saved = read(PUB / "examples.json")
    yan = read(PUB / "historical-reading.json")["yan"]
    checks = [(e["task"], e["output"], e["stop"], e["fact_hash"]) for e in saved]
    checks.extend(
        (
            "yan-" + v["language"] + "-" + v["stage"],
            v["answer_hash"],
            v["stopped"],
            None,
        )
        for v in yan
        if v["arm"] == "whole"
    )
    replay = []
    for name, expected, stop, fact in checks:
        folder = WORK / "replay" / name
        manifest = MODEL / "examples" / name / "input.json"
        if not (folder / "receipt.json").exists():
            write(
                Namespace(
                    input=manifest,
                    device="cpu",
                    arm="whole",
                    model_dir=BASE,
                    output=folder,
                )
            )
        rc = read(folder / "receipt.json")
        if (
            rc["sha256"] != expected
            or rc["stopped"] != stop
            or fact
            and rc["fact_sha256"] != fact
        ):
            raise ValueError("Physical replay mismatch: " + name)
        replay.append(dict(example=name, **rc))
    dump(PUB / "verification.json", dict(real_gradients=checked, cpu_replay=replay))
    print(checked, flush=True)


if __name__ == "__main__":
    main()
