"""Which pictured sources support the same fixed human translation?"""

import numpy as np
import torch
from scripts.prepare_book_rehearsal import WORK, PUB, read, dump, sha
from scripts.train_paired_ink import conditional, material
from scripts.train_ordered_translation import pack
from scripts.write_ordered_translation import restore


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    rows = read(WORK / "generated.json")["rows"]
    net = restore("book-rehearsal", "cpu")
    features, inventories, mappings = material("cpu")
    groups = {}
    matrices = {}
    for domain in ("new-book", "new-flores"):
        for t in (0, 1):
            cases = [r for r in rows if r["group"] == domain and r["t"] == t]
            assert len(cases) == 24 and len({tuple(r["input"]) for r in cases}) == 24
            source = pack([r["input"] for r in cases], "cpu")
            logp = []
            for r in cases:
                target = torch.tensor([r["target"] + [1]]).expand(len(cases), -1)
                score = conditional(
                    net, features, source, target, inventories[t], mappings[t], t
                )
                logp.append(score.double())
            scores = torch.stack(logp)
            truth = scores.diag()
            # Count ties against the true source. The same target and its stop are
            # scored under every source, so its unconditional fluency cancels.
            rivals = scores.clone()
            rivals.fill_diagonal_(-torch.inf)
            ranks = 1 + (rivals >= truth[:, None]).sum(1)
            subset_correct = 0
            for i, r in enumerate(cases):
                nearby = sorted(
                    (j for j in range(len(cases)) if i != j),
                    key=lambda j: (abs(len(r["input"]) - len(cases[j]["input"])), j),
                )[:5]
                subset_correct += int(truth[i] > scores[i, nearby].max())
            # An order control keeps the entire source bag, length and target unchanged.
            reversed_source = pack([r["input"][::-1] for r in cases], "cpu")
            targets = pack([r["target"] + [1] for r in cases], "cpu")
            flipped = conditional(
                net, features, reversed_source, targets, inventories[t], mappings[t], t
            ).double()
            key = domain + ("-en" if t == 0 else "-zh")
            matrices[key] = scores.numpy()
            groups[key] = dict(
                n=len(cases),
                rank1=int((ranks == 1).sum()),
                rank6_length_matched=subset_correct,
                strict_source_over_reverse=int((truth > flipped).sum()),
                ranks=ranks.tolist(),
                mean_log2_posterior_gain=float(
                    (np.log(24) + scores.log_softmax(1).diag().mean()) / np.log(2)
                ),
                mean_order_gain_nats_per_target_glyph=float(
                    (truth - flipped).sum() / targets.count_nonzero()
                ),
            )
    np.savez(WORK / "source-scores.npz", **matrices)
    dump(
        PUB / "conditioning.json",
        dict(
            groups=groups,
            generated=sha(WORK / "generated.json"),
            scores=sha(WORK / "source-scores.npz"),
        ),
    )
    print({k: v["rank1"] for k, v in groups.items()})


if __name__ == "__main__":
    main()
