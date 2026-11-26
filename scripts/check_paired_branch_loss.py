"""Check the prospective branch objective on physical rows without making an optimizer update."""

from hashlib import sha256
from zipfile import ZipFile
import json
import torch

from scripts.train_paired_branch_writer import branch_losses
from scripts.measure_paired_visual_writer import (
    ARCHIVE,
    PUB as OLD,
    base,
    restore,
    unpack,
    pack,
    pictured_input,
    CoveredWriter,
)
from scripts.prepare_opus_paired_supervision import PUB, ROOT


def main():
    torch.set_num_threads(3)
    net = restore("broad", "cpu")
    original = {
        k: sha256(v.detach().numpy().tobytes()).hexdigest()
        for k, v in net.model.state_dict().items()
    }
    audited = base.read(OLD / "verification.json")
    natural = set(base.read(OLD / "provenance.json")["natural"]["audit"])
    selected = [
        i
        for i, ok in zip(audited["ids"], audited["complete"], strict=True)
        if ok and i in natural
    ][:3]
    checks = []
    face = base.Ink(net.palette)
    with ZipFile(ARCHIVE) as z:
        rows = {
            (r["i"], r["t"], r["side"]): r for r in json.loads(z.read("records.json"))
        }
        for t in (0, 1):
            engine = CoveredWriter(t)
            sources, drafts, targets, columns = [], [], [], []
            for i in selected:
                for s in (0, 1):
                    r = rows[i, t, s]
                    img = unpack(
                        z.read(f"{i}/{t}/{s}/source.png"), len(face.encode(r["source"]))
                    )
                    sources.append(net.addresses(img))
                    drafts.append(net.addresses(engine.translate(img)[0]) or [0])
                    targets.append(face.encode(r["ref"]) + [1])
                pos = next(
                    k
                    for k, (a, b) in enumerate(zip(targets[-2], targets[-1]))
                    if a != b
                )
                columns.extend([pos, pos])
            source = pack(sources, "cpu")
            field = (
                (
                    255
                    * net.marker(net.features, source).sigmoid()
                    * source.ne(0).unsqueeze(-1)
                )
                .round()
                .byte()
            )
            target = pack(targets, "cpu")
            col = torch.tensor(columns)
            sm = net.model(
                net.features,
                pictured_input(source, field),
                target,
                net.inv[t],
                t,
                pack(drafts, "cpu"),
                net.local[t],
            )
            wanted = net.local[t][target]
            pos, rank = branch_losses(sm, wanted, col)
            # Independently reconstruct the loss from each literal source/answer corner.
            deltas = []
            self_scores = []
            for j in range(len(selected)):
                n = 2 * j
                at = col[n]
                a, b = wanted[n : n + 2, at]
                x, y = sm[n : n + 2, at]
                self_scores.extend((x[a], y[b]))
                deltas.append(((x[a] - x[b]) + (y[b] - y[a])) * 0.5)
            explicit = (
                torch.stack(
                    [torch.logaddexp(x.new_zeros(()), 2 * (1 - x)) for x in deltas]
                ).mean()
                * 0.5
            )
            torch.testing.assert_close(rank, explicit)
            torch.testing.assert_close(pos, -torch.stack(self_scores).mean())
            shared = torch.zeros(sm.size(-1), requires_grad=True)
            _, shifted = branch_losses(
                (sm.detach() + shared).log_softmax(-1), wanted, col
            )
            grad = torch.autograd.grad(shifted, shared)[0]
            assert grad.abs().max() < 1e-6
            net.model.zero_grad(set_to_none=True)
            (pos + 0.25 * rank).backward()
            valid = [p.grad for p in net.model.parameters() if p.grad is not None]
            assert (
                all(torch.isfinite(v).all() for v in valid)
                and sum(float(v.abs().sum()) for v in valid) > 0
            )
            checks.append(
                dict(
                    direction=t,
                    families=selected,
                    branch_positions=columns,
                    positive=float(pos.detach()),
                    rank=float(rank.detach()),
                    shared_bias_gradient=float(grad.abs().max()),
                    tensors_with_gradient=len(valid),
                )
            )
    assert original == {
        k: sha256(v.detach().numpy().tobytes()).hexdigest()
        for k, v in net.model.state_dict().items()
    }
    base.dump(
        PUB / "loss-check.json",
        dict(
            items=checks,
            initial=base.read(OLD / "writer.json")["fit"],
            optimizer_updates=0,
            objective=base.sha(ROOT / "scripts/train_paired_branch_writer.py"),
            previous=base.sha(ARCHIVE),
        ),
    )
    print(checks, flush=True)


if __name__ == "__main__":
    main()
