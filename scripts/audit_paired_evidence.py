"""Audit the joint mass of the reference ink, question contrasts and alternate readings."""

from collections import defaultdict
import gzip
import json
import math
import numpy as np
import torch
from torch.nn import functional as F
from scripts.prepare_paired_evidence import (
    WORK,
    PUB,
    ROOT,
    read,
    sha,
    dump,
    hash_text,
    clean,
)
from scripts.train_paired_evidence import (
    Pairs,
    Corpus,
    gpu_owner,
    load_model,
    byte_features,
)
from scripts.write_paired_evidence import reader
from scripts.train_visual_cursor import pad, base_batch
from scripts.evaluate_visual_cursor import normalize
from scripts.prepare_visual_sequence_corpus import Rasterizer
from ilm.visual_lm.paired_evidence import compatible
from ilm.visual_lm.book_prefix import observations
from scripts.evaluate_visual_discourse import packed


def details(arm):
    return json.loads(gzip.decompress((PUB / (arm + "-qa.json.gz")).read_bytes()))


def alternate():
    path = ROOT / "data/raw/coqa/dev-official.json"
    if sha(path) != "dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a":
        raise ValueError("Changed dialogue references")
    key = {}
    for doc in read(path)["data"]:
        page = hash_text(clean(doc["story"]))
        for q in doc["questions"]:
            refs = set()
            for versions in [doc["answers"], *doc["additional_answers"].values()]:
                refs.update(
                    normalize(v["input_text"])
                    for v in versions
                    if v["turn_id"] == q["turn_id"]
                )
            key[page, clean(q["input_text"])] = refs
    return key


def comparisons():
    truth = alternate()
    old = details("answer")
    new = details("evidence")
    result = {}
    for arm in ("cursor", "answer", "evidence", "byte-evidence"):
        g = [r for r in details(arm) if r["kind"] == "coqa" and r["test"] == "human"]
        ok = defaultdict(list)
        absent = answered = 0
        pages = {r["doc"]: r["source"] for r in read(WORK / "held.json")}
        for r in g:
            refs = truth[r["doc"], r["question"]]
            correct = r["stopped"] and normalize(r["output"]) in refs
            ok[r["doc"]].append(correct)
            if all(t not in normalize(pages[r["doc"]]) for t in refs):
                absent += 1
                answered += correct
        result[arm] = dict(
            any_reference=sum(sum(v) for v in ok.values()),
            both=sum(all(v) for v in ok.values()),
            absent_all=absent,
            absent_any_correct=answered,
        )
    a = {}
    b = {}
    for src, groups in ((old, a), (new, b)):
        for r in src:
            if r["test"] == "human":
                groups.setdefault(r["doc"], []).append(r["correct"])
    wins = sum(all(b[k]) and not all(a[k]) for k in a)
    losses = sum(all(a[k]) and not all(b[k]) for k in a)
    n = wins + losses
    prob = min(
        1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    )
    dump(
        PUB / "paired-audit.json",
        dict(
            dialogue=result,
            discordant_passages=dict(
                marker_only=wins, answer_only=losses, two_sided_exact_p=prob
            ),
        ),
    )


@torch.inference_mode()
def main():
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    comparisons()
    data = Pairs()
    old = Corpus()
    held = read(WORK / "held.json")
    raster = Rasterizer()
    lookup = {bytes(g): i for i, g in enumerate(data.a["tiles"]) if i >= 2}
    tests = []
    excluded = 0
    for d in held:
        try:
            targets = [
                [lookup[raster.tile(c)] for c in r["refs"][0]] + [1] for r in d["rows"]
            ]
        except (ValueError, KeyError):
            excluded += 1
            continue
        if max(map(len, targets)) > 112:
            excluded += 1
            continue
        for i, r in enumerate(d["rows"]):
            for j, y in enumerate(targets):
                tests.append((d["doc"], d["kind"], i, j, r["q"], d["s"], y))
    with gpu_owner():
        for arm in ("answer", "evidence", "byte-evidence"):
            byte = arm == "byte-evidence"
            x = (
                byte_features(data.a["chars"]).cuda()
                if byte
                else torch.from_numpy(
                    observations({"tiles": data.a["tiles"]}, "ink")
                ).cuda()
            )
            base = load_model("byte-pair" if byte else "ink-pair", "cuda")
            bk = base.encode(x)
            net = reader(arm, "cuda")
            bank = net.ink(x)
            reported = {
                (r["doc"], r["idx"]): r for r in details(arm) if r["test"] == "human"
            }
            scores = defaultdict(list)
            rows = []
            for k in range(0, len(tests), 24):
                g = tests[k : k + 24]
                qs, ss, ys = zip(*(r[4:] for r in g), strict=True)
                q, s, y = [torch.as_tensor(pad(v), device="cuda") for v in (qs, ss, ys)]
                bi, bp = [
                    torch.as_tensor(z, device="cuda")
                    for z in base_batch([p + u for p, u in zip(ss, qs)], ys)
                ]
                blp = base.log_prob(bk, bi, old.n, bp)
                spans, _, h, gate = net.read(bank, q, s)
                hist = torch.cat(
                    (net.begin.expand(len(g), 1, -1), bank[y[:, :-1].long()]), 1
                )
                z, _ = net.gru(hist, h[None].contiguous())
                lp = (
                    net.glyph_scores(z, bank, old.n)
                    .gather(-1, (y.long() - 1).clamp(0, old.n - 2)[:, :, None])
                    .squeeze(-1)
                    .masked_fill(y >= old.n, -1e9)
                )
                mix = net.free_mix(z).squeeze(-1)
                free = (
                    torch.logaddexp(F.logsigmoid(mix) + lp, F.logsigmoid(-mix) + blp)
                    * (y > 0)
                ).sum(-1)
                sp = spans.masked_fill(~compatible(s, y), -1e9).flatten(1).logsumexp(-1)
                atom = torch.logaddexp(
                    F.logsigmoid(gate) + sp, F.logsigmoid(-gate) + free
                )
                for r, v, span, gg in zip(
                    g, atom.tolist(), sp.tolist(), gate.tolist(), strict=True
                ):
                    doc, kind, i, j = r[:4]
                    scores[doc].append(v)
                    if i == j:
                        ex = reported[doc, i]
                        exact = ex["stopped"] and ex["glyphs"] == r[-1][:-1]
                        if v > math.log(0.5) + 1e-5 and not exact:
                            raise ValueError(
                                "Majority certificate disagrees with physical decode"
                            )
                        rows.append(
                            dict(
                                kind=kind,
                                doc=doc,
                                idx=i,
                                log_mass=v,
                                log_span=span,
                                gate_logit=gg,
                                physical_exact=exact,
                            )
                        )
            summary = {}
            for kind in ("coqa", "squad", "cmrc", "drcd"):
                subset = [r for r in rows if r["kind"] == kind]
                docs = {r["doc"] for r in subset}
                margins = [scores[d] for d in docs]
                summary[kind] = dict(
                    n=len(subset),
                    pairs=len(docs),
                    both_rank=sum(aa > ab and bb > ba for aa, ab, ba, bb in margins),
                    assigned=sum(
                        aa + bb > ab + ba + 1e-6 for aa, ab, ba, bb in margins
                    ),
                    majority=sum(r["log_mass"] > math.log(0.5) for r in subset),
                    supported=sum(r["log_span"] > -1e8 for r in subset),
                    median_gold_probability=float(
                        np.median([math.exp(r["log_mass"]) for r in subset])
                    ),
                )
            packed(PUB / (arm + "-atoms.json.gz"), rows)
            dump(
                PUB / (arm + "-atoms.json"),
                dict(summary=summary, excluded_pairs=excluded),
            )
            print(arm, summary, flush=True)


if __name__ == "__main__":
    main()
