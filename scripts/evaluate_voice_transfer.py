"""New deterministic bidirectional-role test of English voice and polarity in fixed weights."""

from collections import Counter
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_translation_paraphrases import (
    BASE,
    OLD,
    FIRST,
    PUB,
    WORK,
    read,
    dump,
    key,
    sha,
)
from scripts.prepare_proposition_translation import (
    NAMES,
    VERBS,
    capitalize,
    hashed,
    Ink,
)
from scripts.prepare_bilingual_ink import source as opus
from scripts.train_ordered_translation import pack
from scripts.write_ordered_translation import restore, stopped


def groups():
    participles = (
        "helped",
        "visited",
        "thanked",
        "called",
        "invited",
        "followed",
        "seen",
        "found",
        "hugged",
        "praised",
    )
    for i, (a, x) in enumerate(NAMES):
        for j, (b, y) in enumerate(NAMES[:i]):
            if int(hashed(f"proposition-names/{j}/{i}")[:4], 16) % 5:
                continue
            for (past, base, zh, stem), part in zip(VERBS, participles, strict=True):
                group = []
                for subject, agent, patient, obj in ((a, x, b, y), (b, y, a, x)):
                    for negative in (False, True):
                        target = (
                            agent + ("没有" + stem if negative else zh) + obj + "。"
                        )
                        active = (
                            f"{subject} did not {base} {patient}."
                            if negative
                            else f"{subject} {past} {patient}."
                        )
                        passive = f"{patient} was {'not ' if negative else ''}{part} by {subject}."
                        for voice, src in (("active", active), ("passive", passive)):
                            group.append(
                                dict(
                                    voice=voice,
                                    negative=negative,
                                    source=capitalize(src),
                                    reference=target,
                                )
                            )
                yield group


def unseen(all_groups):
    wanted = [
        {key(r[k]) for group in all_groups for r in group}
        for k in ("source", "reference")
    ]
    seen = [set(), set()]

    def inspect(s, t):
        if len(s) <= 96 and (k := key(s)) in wanted[t]:
            seen[t].add(k)

    for name in ("train", "news-train"):
        for r in read(OLD / (name + ".json")):
            for t, s in enumerate(r["text"]):
                inspect(s, t)
    for z in opus("train"):
        for t, s in enumerate(z):
            inspect(s, t)
    lu = read(BASE / "readback.json")

    def check_cells(c, t):
        inspect("".join(lu[str(i)] for i in c), t)

    for r in read(FIRST / "contrasts.json"):
        for z in r["cells"]:
            for t, c in enumerate(z):
                check_cells(c, t)
    for r in read(WORK / "examples.json"):
        for t in (0, 1):
            for c in [r["target"][t], *r["variants"][t]]:
                check_cells(c, t)
    for r in read(WORK / "generated.json")["rows"]:
        inspect(r["source"], 1 - r["t"])
        inspect(r["ref"], r["t"])
    return [
        g
        for g in all_groups
        if all(
            key(r["source"]) not in seen[0] and key(r["reference"]) not in seen[1]
            for r in g
        )
    ]


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    candidates = list(groups())
    kept = unseen(candidates)
    if not kept:
        raise ValueError("No complete families remain after exclusions")
    rows = [r for g in kept for r in g]
    ink = Ink(np.load(BASE / "palette.npy"))
    source = [ink.encode(r["source"]) for r in rows]
    refs = [ink.encode(r["reference"]) for r in rows]
    if ink.bad:
        raise ValueError(ink.bad)
    f = torch.tensor(observations({"tiles": ink.array()}, "ink"))
    inventory = torch.tensor(read(BASE / "inventories.json")[1])
    scores = {}
    for m in ("canonical", "varied"):
        net = restore("paraphrase-" + m, "cpu")
        count = Counter()
        ok = []
        for i in range(0, len(rows), 32):
            ys = net.generate(f, pack(source[i : i + 32], "cpu"), inventory, 1)
            for j, y in enumerate(ys, i):
                cells, stop = stopped(y)
                yes = stop and cells == refs[j]
                rows[j][m] = dict(cells=cells, stop=stop, exact=yes)
                ok.append(yes)
                count[rows[j]["voice"]] += int(yes)
                count["negative" if rows[j]["negative"] else "positive"] += int(yes)
        count["all_eight"] = sum(all(ok[i : i + 8]) for i in range(0, len(ok), 8))
        scores[m] = count
    dump(WORK / "voice-outputs.json", rows)
    report = dict(
        initial_families=len(candidates),
        families=len(kept),
        requests=len(rows),
        scores=scores,
        rows_hash=sha(WORK / "voice-outputs.json"),
        checkpoint={
            m: sha(BASE.parent / "translation_paraphrases" / m / "weights.safetensors")
            for m in scores
        },
    )
    dump(PUB / "voice.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
