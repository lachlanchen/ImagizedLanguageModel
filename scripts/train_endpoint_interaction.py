"""Fit matched endpoint models and score self-written ink on a second reserved split."""

from math import comb

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.endpoint_interaction import EndpointWriter
from scripts.reserve_endpoint_pages import WORK, PUB, MODEL, main as reserve
from scripts.prepare_competing_facts import PAIRED, ROOT, dump, read, sha
from scripts.prepare_competing_facts import PUB as previous_pub, WORK as previous_work
from scripts.evaluate_competing_facts import dataset, assess
from scripts.evaluate_visual_discourse import packed, unpack
from scripts.train_competing_facts import fit
from scripts.train_paired_evidence import gpu_owner
from scripts.write_paired_evidence import reader


def restored(mode, device):
    net = EndpointWriter(mode).to(device)
    file = MODEL / (mode + ".safetensors")
    if sha(file) != read(PUB / (mode + ".json"))["weights"]:
        raise ValueError("Endpoint weights changed")
    net.load_state_dict(load_file(str(file), device=device))
    return net.eval()


def results(output):
    scores = {}
    pages = {}
    for name, rows in output.items():
        grouped = {}
        for r in rows:
            grouped.setdefault((r["kind"], r["doc"]), []).append(bool(r["correct"]))
        assert all(len(v) == 2 for v in grouped.values()) and len(grouped) == 128
        pages[name] = {k: all(v) for k, v in grouped.items()}
        groups = {}
        for lang, kinds in (
            ("en", ("squad", "coqa")),
            ("zh", ("cmrc", "drcd")),
            ("all", ("squad", "coqa", "cmrc", "drcd")),
        ):
            sub = [v for (k, _), v in grouped.items() if k in kinds]
            groups[lang] = dict(
                pages=len(sub),
                both=sum(all(v) for v in sub),
                correct=sum(sum(v) for v in sub),
            )
        scores[name] = groups
    changes = {}
    for control in ("parent", "additive"):
        gain = sum(b and not pages[control][k] for k, b in pages["product"].items())
        loss = sum(not b and pages[control][k] for k, b in pages["product"].items())
        n = gain + loss
        p = min(1.0, 2 * sum(comb(n, i) for i in range(min(gain, loss) + 1)) / 2**n)
        changes[control] = dict(gain=gain, loss=loss, paired_exact_p=p)
    adopt = all(
        scores["product"][k]["both"] > scores[c][k]["both"]
        for c in ("parent", "additive")
        for k in ("en", "zh")
    )
    return dict(scores=scores, changes=changes, adopt=adopt)


def main():
    reserve()
    meta = read(previous_pub / "corpus.json")
    if (
        sha(PAIRED / "corpus.npz") != meta["parents"]
        or sha(previous_work / "train.json") != meta["files"]["train.json"]
    ):
        raise ValueError("Training data changed")
    if (
        sha(ROOT / "references/endpoint_interaction_protocol.md")
        != read(PUB / "split.json")["protocol"]
    ):
        raise ValueError("Reserved protocol changed")
    groups = read(previous_work / "train.json")
    with np.load(PAIRED / "corpus.npz") as z:
        arrays = {k: z[k] for k in ("p", "po", "q", "qo", "tiles", "chars")}
    rows, enc = dataset(
        WORK / "held.json", read(PUB / "split.json")["sha256"], statements=False
    )
    rows = [r for r in rows if r["test"] == "human"]
    with gpu_owner():
        parent = reader("evidence", "cuda")
        for mode in ("additive", "product"):
            torch.manual_seed(291408)
            net = EndpointWriter(mode).cuda()
            missing, extra = net.load_state_dict(parent.state_dict(), strict=False)
            assert (
                set(missing)
                == {"gain", "left.weight", "left.bias", "right.weight", "right.bias"}
                and not extra
            )
            fit(
                mode,
                groups,
                arrays,
                None,
                model_dir=MODEL,
                metrics_dir=PUB,
                initial=net,
                resume_dir=WORK,
            )
        output = {}
        for mode in ("parent", "additive", "product"):
            path = PUB / (mode + "-outputs.json.gz")
            receipt = PUB / (mode + "-evaluation.json")
            if receipt.exists():
                if sha(path) != read(receipt)["sha256"]:
                    raise ValueError("Evaluation changed")
                output[mode] = unpack(path)
                continue
            net = parent if mode == "parent" else restored(mode, "cuda")
            detail = assess(
                mode, rows, enc, supplied_reader=net, write_statements=False
            )
            packed(path, detail)
            dump(receipt, dict(sha256=sha(path), questions=len(detail)))
            output[mode] = detail
    metrics = results(output)
    dump(PUB / "comparison.json", metrics)
    print(metrics, flush=True)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
