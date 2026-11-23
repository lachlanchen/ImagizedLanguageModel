"""Measure rival separation, unrelated human questions and previously unseen historical name pixels."""

from collections import Counter, defaultdict
import argparse
import fcntl
import json
import os
import torch

from ilm.visual_lm.bound_fact_reader import (
    BoundFactReader,
    best_bounds,
    features_batch,
    render_fact,
)
from ilm.visual_lm.ordered_page_features import contextualize
from ilm.visual_lm.role_conditioned_span import ParticipantReader
from ilm.visual_lm.pictured_relation import PicturedRelation
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.evaluate_participant_margin import previous
from scripts.prepare_participant_margin import OUT, PUBLIC
from scripts.prepare_visual_facts import Face, FONTS, ROOT, WORK, write_json
from scripts.train_participant_margin import read
from scripts.train_visual_facts import READER


def broad_cases(face):
    ledger = json.loads((WORK / "transfer_questions.json").read_text())
    from scripts.span_reservation import reservation
    if any(c["group"]=="drcd" and c["key"] in reservation().by_question for c in ledger):
        raise ValueError("Archived transfer questions include the protected DRCD split")
    need = {(c["group"], c["key"]) for c in ledger}
    result = {}
    for group, file in [
        ("cmrc", "data/raw/cmrc2018/cmrc2018_dev.json"),
        ("drcd", "data/raw/drcd/DRCD_dev.json"),
    ]:
        for article in json.loads((ROOT / file).read_text())["data"]:
            for para in article["paragraphs"]:
                for qa in para["qas"]:
                    key = (group, qa["id"])
                    if key not in need:
                        continue
                    p = para["context"][:416]
                    refs = [
                        face(a["text"])
                        for a in qa["answers"]
                        if a["text"] in p and 0 < len(a["text"]) <= 48
                    ]
                    result[key] = ((face(p), face(qa["question"])), refs)
    assert len(result) == len(need) == 256
    return [result[(v["group"], v["key"])] for v in ledger]


@torch.inference_mode()
def main(ablation=False):
    name = "likelihood" if ablation else "matched"
    dest = PUBLIC / ("likelihood_transfer.json" if ablation else "transfer.json")
    if dest.exists():
        raise FileExistsError(dest)
    face = Face((FONTS[0], 2))
    base = BoundFactReader(READER, WORK / "genuine", "cuda")
    channel = PicturedRelation(ROOT / "models/pictured_relation/pictured-relation.npz")
    model = ParticipantReader(base, channel, OUT / (name + ".safetensors"))
    report = {}
    broad = broad_cases(face)
    imgs = [
        load_streams(WORK / f"alias_{s}.npz")[0]
        for s in ("pages", "questions", "reference")
    ]
    aliases = [((p, q), [ref]) for p, q, ref in zip(*imgs, strict=True)]
    for group, rows in [("general_direct", broad), ("historic_names", aliases)]:
        c = Counter()
        for i in range(0, len(rows), 32):
            ps = [x[0] for x in rows[i : i + 32]]
            values = contextualize(base.reader, ps)
            for arm, fields in [
                ("previous", previous(base, channel, ps, values)),
                (name, model.from_features(ps, values)),
            ]:
                for (_, refs), (s, r, o) in zip(rows[i : i + 32], fields, strict=True):
                    y = (
                        o
                        if group == "general_direct"
                        else render_fact(base.frames[r], s, o)
                    )
                    c[arm] += y in refs
        report[group] = dict(c, n=len(rows))
    assert report["general_direct"]["previous"] == 147
    cases = read("new")
    stats = defaultdict(lambda: defaultdict(Counter))
    for i in range(0, len(cases), 32):
        chunk = cases[i : i + 32]
        ps = [(face(v["p"]), face(v["q"])) for v in chunk]
        data = features_batch(contextualize(base.reader, ps), "cuda")
        roles, old = base.head(*data)
        sb = best_bounds(old[:, :, :2]).tolist()
        chosen = [
            channel.choose(q, p[128 * a : 128 * b], r)[0]
            for (p, q), (a, b), r in zip(ps, sb, roles.argmax(1).tolist(), strict=True)
        ]
        new = model.head(
            data[1], data[2], torch.tensor(chosen, device="cuda"), old[:, :, 2:]
        )
        for arm, scores in [("previous", old[:, :, 2:]), (name, new)]:
            picks = best_bounds(scores).tolist()
            scores = scores.cpu()
            for n, (v, (a, b)) in enumerate(zip(chunk, picks, strict=True)):
                p = v["p"]
                t = p.index(v["o"])
                r = p.index(v["rival"])
                tend = t + len(v["o"])
                rend = r + len(v["rival"])
                gap = float(
                    scores[n, t, 0]
                    + scores[n, tend - 1, 1]
                    - scores[n, r, 0]
                    - scores[n, rend - 1, 1]
                )
                correct = (t, tend) == (a, b)
                c = stats[arm][v["group"]]
                c.update(
                    n=1,
                    target_beats_rival=gap > 0,
                    full_span_correct=correct,
                    wins_pair_wrong_full=gap > 0 and not correct,
                    reference_margin_below_one=gap < 1,
                )
    report["span_gaps"] = stats
    write_json(dest, report)
    print(report, flush=True)


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU0 required")
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(3)
        parser = argparse.ArgumentParser()
        parser.add_argument("--likelihood-only", action="store_true")
        main(parser.parse_args().likelihood_only)
