"""Score complete supplied alternatives under real and neutral pictured premises."""

import tarfile
import xml.etree.ElementTree as ET
import numpy as np
import torch
from scripts.prepare_causal_book_words import PUB, ROOT, old
from scripts.train_causal_book_words import gpu_owner, configure_cuda_cli
from ilm.visual_lm.passage_selector import Print, VisualBank
from ilm.visual_lm.causal_book_words import CausalBookWriter
from scripts.score_passage_selector import bounds

DIR = ROOT / "data/raw/passage-copa"
SOURCES = {
    "original.tgz": (
        "https://asgordon.github.io/downloads/COPA-resources.tgz",
        "5145348834d2081ad90da0397d1db3d70fa044e506bd8ce224194d24b04cdbbe",
    ),
    "balanced.xml": (
        "https://raw.githubusercontent.com/Balanced-COPA/Balanced-COPA/92c2ec6d792b3e43a741d4e94ceb731b25f9a719/balanced-copa-dev-all.xml",
        "2091dce8a2923b8530804644780ebe01fd1ae30342eb2562a3821786274b5b25",
    ),
}
LOWER = {
    "The",
    "A",
    "An",
    "He",
    "She",
    "His",
    "Her",
    "They",
    "Their",
    "It",
    "Its",
    "My",
    "We",
    "Our",
    "This",
    "That",
    "There",
}


def questions():
    for name, (url, h) in SOURCES.items():
        old.download(url, DIR / name, h)
    with tarfile.open(DIR / "original.tgz") as archive:
        root = ET.fromstring(
            archive.extractfile("COPA-resources/datasets/copa-test.xml").read()
        )
    out = {}
    for split, tree in [
        ("original", root),
        ("balanced", ET.parse(DIR / "balanced.xml").getroot()),
    ]:
        rows = []
        for r in tree:
            rows.append(
                dict(
                    id=int(r.get("id")),
                    ask=r.get("asks-for"),
                    label=int(r.get("most-plausible-alternative")) - 1,
                    premise=r.findtext("p"),
                    choices=[r.findtext("a1").strip(), r.findtext("a2").strip()],
                )
            )
        out[split] = sorted(rows, key=lambda r: r["id"])
    assert len(out["original"]) == 500 and len(out["balanced"]) == 1000
    for a, b in zip(out["balanced"][:500], out["balanced"][500:], strict=True):
        assert (
            b["id"] == a["id"] + 1000
            and set(a["choices"]) == set(b["choices"])
            and a["choices"][a["label"]] != b["choices"][b["label"]]
            and a["ask"] == b["ask"]
        )
    return out


def design(rows):
    face = Print()
    ink = old.base.Ink()
    forms = [()]
    index = {(): 0}
    inputs = []
    owners = []

    def optical(text):
        ids = ink.encode(text)
        words, gaps = face.scan(ink.array()[ids])
        for w in words:
            if w not in index:
                index[w] = len(forms)
                forms.append(w)
        return [index[w] for w in words], gaps

    lengths = np.zeros((len(rows), 2, 2), int)
    for i, row in enumerate(rows):
        connector = " because " if row["ask"] == "cause" else ", so "
        for c, choice in enumerate(row["choices"]):
            first = choice.split()[0]
            if first in LOWER:
                choice = choice[0].lower() + choice[1:]
            y, yg = optical(choice)
            for neutral in (0, 1):
                prefix = "" if neutral else row["premise"].rstrip().rstrip(".?!")
                words, gaps = optical(prefix + connector)
                for k, word in enumerate(y):
                    inputs.append((list(words), list(gaps), word))
                    owners.append((i, c, neutral))
                    words.append(word)
                    gaps.append(yg[k])
                lengths[i, c, neutral] = len(y)
    return face, forms, inputs, np.array(owners).T, lengths


def mass(net, bank, src, marks, target):
    q, a, g = net.observe(
        bank, src, marks, torch.zeros(len(src), dtype=torch.long, device=src.device)
    )
    copy = (a * (src == target[:, None])).sum(1) * g
    sp = net.spell_logprob(q, bank.codes[target], bank.lengths[target])
    return torch.logaddexp(copy.log(), torch.log1p(-g) + sp)


@torch.inference_mode()
def main():
    configure_cuda_cli()
    data = questions()
    scorefile = PUB / "copa_scores.npz"
    if scorefile.exists():
        raise SystemExit("Frozen causal-choice scores exist")
    arrays = {}
    with gpu_owner():
        for split, rows in data.items():
            face, forms, inputs, owners, lengths = design(rows)
            bank = VisualBank(face, forms, "cuda")
            for arm in ("dense", "endpoint"):
                net = CausalBookWriter(arm, "cuda").net
                values = []
                for start in range(0, len(inputs), 160):
                    batch = inputs[start : start + 160]
                    width = max(len(e[0]) for e in batch)
                    assert width <= 96
                    ss = np.zeros((len(batch), width), np.int64)
                    mm = np.zeros_like(ss)
                    yy = []
                    for j, (seq, gaps, target) in enumerate(batch):
                        ss[j, -len(seq) :] = seq
                        mm[j, -len(seq) :] = gaps
                        yy.append(target)
                    s, m, y = [torch.as_tensor(a, device="cuda") for a in (ss, mm, yy)]
                    assert bank.lengths[y].min() > 0
                    values.extend(mass(net, bank, s, m, y).tolist())
                scores = np.zeros_like(lengths, dtype=np.float64)
                np.add.at(scores, tuple(owners), values)
                arrays[split + "_" + arm] = scores / lengths
    np.savez_compressed(scorefile, **arrays)
    out = {}
    with np.load(scorefile) as saved:
        for split, rows in data.items():
            label = np.array([r["label"] for r in rows])
            for arm in ("dense", "endpoint"):
                a = saved[split + "_" + arm]
                values = {}
                for name, x in [
                    ("conditional", a[:, :, 0]),
                    ("neutral", a[:, :, 1]),
                    ("calibrated", a[:, :, 0] - a[:, :, 1]),
                ]:
                    difference = (x[:, 0] - x[:, 1]) * (1 - 2 * label)
                    ok = difference > 1e-7
                    groups = np.arange(len(ok)) % (
                        500 if split == "balanced" else len(ok)
                    )
                    values[name] = dict(
                        correct=int(ok.sum()),
                        ties=int((np.abs(difference) <= 1e-7).sum()),
                        n=len(ok),
                        accuracy_ci=bounds(ok, np.zeros(len(ok)), groups),
                    )
                    if split == "balanced":
                        values[name].update(
                            both=int((ok[:500] & ok[500:]).sum()),
                            interaction=int(
                                ((difference[:500] + difference[500:]) > 1e-7).sum()
                            ),
                        )
                out[split + "_" + arm] = values
    old.base.dump(
        PUB / "copa.json",
        dict(
            scores=old.base.sha(scorefile),
            sources=SOURCES,
            primary="calibrated",
            outcomes=out,
        ),
    )
    print({k: v["calibrated"] for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
