"""Isolate target location from the opposite boundary in held physical interval scores."""

from collections import Counter
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.evaluate_competing_facts import dataset, positions
from scripts.prepare_competing_facts import PUB, dump
from scripts.train_visual_cursor import pad
from scripts.train_paired_evidence import gpu_owner
from scripts.write_paired_evidence import reader
from scripts.evaluate_visual_discourse import packed, unpack


@torch.inference_mode()
def main():
    rows, enc = dataset()
    rows = [r for r in rows if r["test"] == "human"]
    for r in rows:
        r["targets"] = positions(r["s"], r["refs"], enc)
    features = torch.from_numpy(
        observations(
            {"tiles": np.array([np.frombuffer(x, np.uint8) for x in enc.tiles])}, "ink"
        )
    ).cuda()
    report = {}
    for arm in ("parent", "likelihood", "contrast"):
        net = reader("evidence" if arm == "parent" else arm, "cuda")
        free = {
            r["key"]: r
            for r in unpack(PUB / (arm + "-outputs.json.gz"))
            if r["test"] == "human"
        }
        bank = net.ink(features)
        detail = []
        count = Counter()
        for i in range(0, len(rows), 16):
            rs = rows[i : i + 16]
            q, s = [
                torch.as_tensor(pad([r[k] for r in rs]), device="cuda").long()
                for k in ("q", "s")
            ]
            vals = net.read(bank, q, s)[0].cpu().numpy()
            for r, mat in zip(rs, vals, strict=True):
                if not r["targets"]:
                    continue
                mat = mat[: len(r["s"])]
                targets = set(r["targets"])
                starts = {p for p, l in targets}
                ends = {p + l for p, l in targets}
                grid = np.arange(len(mat))[:, None]
                end = grid + np.arange(mat.shape[1])[None]
                flags = {
                    "unrestricted": np.ones_like(mat, dtype=bool),
                    "given_start": np.isin(grid, list(starts)),
                    "given_end": np.isin(end, list(ends)),
                }
                row = dict(
                    key=r["key"], kind=r["kind"], actual=free[r["key"]]["correct"]
                )
                for name, mask in flags.items():
                    p, l = np.unravel_index(
                        np.where(mask & (end < len(mat)), mat, -np.inf).argmax(),
                        mat.shape,
                    )
                    row[name] = (p, l) in targets
                count.update(
                    {
                        k: int(row[k])
                        for k in ("actual", "unrestricted", "given_start", "given_end")
                    }
                )
                count["eligible"] += 1
                count["neither_boundary_suffices"] += int(
                    not row["given_start"] and not row["given_end"]
                )
                detail.append(row)
        packed(PUB / (arm + "-boundaries.json.gz"), detail)
        report[arm] = dict(count)
    dump(PUB / "boundaries.json", report)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    with gpu_owner():
        main()
