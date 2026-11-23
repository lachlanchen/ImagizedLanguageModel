"""Score actual emitted glyphs and entire exchange groups on fixed image inputs."""

from collections import Counter, defaultdict
import argparse
import fcntl
import json
import os
import numpy as np
import torch
from ilm.visual_lm.bound_fact_reader import (
    BoundFactReader,
    best_bounds,
    features_batch,
    render_fact,
)
from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.fact_realizer import FactRealizer
from ilm.visual_lm.role_conditioned_span import ParticipantReader
from ilm.visual_lm.pictured_relation import PicturedRelation
from scripts.evaluate_grounded_propositions import store_streams
from scripts.prepare_participant_margin import OUT, PUBLIC
from scripts.prepare_visual_facts import ROOT, WORK, Face, FONTS, RELATIONS, write_json
from scripts.train_fact_realizer import target
from scripts.train_participant_margin import read, picture_cache
from scripts.train_visual_facts import rows, READER
from scripts.train_pointing_causal_writer import sha


def diagnostic():
    src = json.loads(
        (ROOT / "references/evidence/pictured_relation_20261123/songs.json").read_text()
    )
    return [
        dict(
            group="song",
            board="diagnostic",
            p=v["page"],
            q=v["question"],
            **v["reference"],
        )
        for v in src
    ]


def save_png(path, buf):
    write_strip(np.frombuffer(buf, np.uint8).reshape(-1, 128), path)


def previous(base, channel, inputs, values):
    _, page, mask = data = features_batch(values, base.device)
    roles, bounds = base.head(*data)
    ss = best_bounds(bounds[:, :, :2]).tolist()
    oo = best_bounds(bounds[:, :, 2:]).tolist()
    result = []
    for (p, q), (a, b), (c, d), r in zip(
        inputs, ss, oo, roles.argmax(1).tolist(), strict=True
    ):
        s = p[128 * a : 128 * b]
        r, _ = channel.choose(q, s, r)
        result.append((s, r, p[128 * c : 128 * d]))
    return result


@torch.inference_mode()
def evaluate(ablation=False):
    dest = PUBLIC / ("ablation.json" if ablation else "results.json")
    if dest.exists():
        raise FileExistsError(dest)
    examples = PUBLIC / "examples"
    examples.mkdir(exist_ok=True)
    base = BoundFactReader(READER, WORK / "genuine", "cuda")
    channel = PicturedRelation(ROOT / "models/pictured_relation/pictured-relation.npz")
    writer = FactRealizer(WORK / "writer", "cuda")
    face = Face((FONTS[0], 2))
    data = rows("held") + read("new") + diagnostic()
    pairs = [(face(v["p"]), face(v["q"])) for v in data]
    features = picture_cache(base.reader, face, data)
    result = {}
    ledgers = {}
    arms = ("likelihood",) if ablation else ("previous", "permuted", "matched")
    for arm in arms:
        model = (
            None
            if arm == "previous"
            else ParticipantReader(base, channel, OUT / (arm + ".safetensors"))
        )
        if (
            model
            and sha(OUT / (arm + ".safetensors"))
            != json.loads((PUBLIC / (arm + "_fit.json")).read_text())["weights"]
        ):
            raise ValueError("Checkpoint differs")
        ledger = []
        totals = defaultdict(Counter)
        boards = defaultdict(lambda: defaultdict(list))
        streams = [[], []]
        stops = [[], []]
        for k in range(0, len(data), 32):
            ps = pairs[k : k + 32]
            vv = features[k : k + 32]
            fields = (
                previous(base, channel, ps, vv)
                if model is None
                else model.from_features(ps, vv)
            )
            seq = [
                writer.answer_fields(fields, order) for order in ("subject", "object")
            ]
            for j, ((s, r, o), x) in enumerate(
                zip(fields, data[k : k + 32], strict=True)
            ):
                group = x["group"]
                c = totals[group]
                c["inputs"] += 1
                hit = o == face(x["o"])
                full = render_fact(base.frames[r], s, o) == face(target(x, 0))
                c.update(
                    object=hit,
                    subject=s == face(x["s"]),
                    role=RELATIONS[r] == x["r"],
                    induced=full,
                )
                if "rival" in x:
                    c["rival_selected"] += o == face(x["rival"])
                good = []
                for order in range(2):
                    raw, stop = seq[order][j]
                    streams[order].append(raw)
                    stops[order].append(stop)
                    good.append(bool(stop and raw == face(target(x, order))))
                    c["stops"] += bool(stop)
                both = all(good)
                c["both"] += both
                board = x.get(
                    "board", str(int(x["key"]) // 4) if group == "notes" else None
                )
                if group in ("new_source", "clause_order", "notes"):
                    boards[group][board].append((hit, full, both))
                index = k + j
                ledger.append(
                    dict(
                        i=index,
                        group=group,
                        board=board,
                        object=hit,
                        induced=full,
                        both=both,
                    )
                )
                # The previously failed song and the first unseen source are chosen by order.
                if index in (1280, 1281, 1282, 1283, len(data) - 2):
                    name = f"{arm}_{index}"
                    if arm == "matched":
                        for suf, buf in [
                            ("page", ps[j][0]),
                            ("question", ps[j][1]),
                            ("reference", face(target(x, 1))),
                        ]:
                            save_png(examples / f"{name}_{suf}.png", buf)
                    save_png(examples / f"{name}_answer.png", seq[1][j][0])
        for group, bs in boards.items():
            expected = 8 if group == "clause_order" else 4
            assert all(len(z) == expected for z in bs.values())
            totals[group].update(boards=len(bs))
            for n, label in enumerate(("objects", "induced", "both")):
                totals[group]["all_" + label] = sum(
                    all(v[n] for v in z) for z in bs.values()
                )
        for order, suffix in enumerate(("subject", "object")):
            store_streams(OUT / f"{arm}_{suffix}.npz", streams[order], stops[order])
        result[arm] = {g: dict(v) for g, v in totals.items()}
        ledgers[arm] = ledger
        write_json(OUT / (arm + "_ledger.json"), ledger)
        print(arm, result[arm], flush=True)
    if ablation:
        write_json(
            dest,
            dict(
                arms=result,
                raw_sha256={p.name: sha(p) for p in OUT.glob("likelihood_*.npz")},
            ),
        )
        return
    before = result["previous"]
    after = result["matched"]
    if [before[g]["both"] for g in ("cmrc", "drcd", "notes", "duie")] != [
        20,
        14,
        18,
        892,
    ]:
        raise ValueError("Previous end-to-end scores do not reproduce")
    checks = dict(
        human=after["cmrc"]["induced"] + after["drcd"]["induced"] >= 36,
        notes=after["notes"]["induced"] >= 17,
        duie=after["duie"]["induced"] >= 930,
    )
    for group in ("new_source", "clause_order"):
        checks[group] = after[group]["all_induced"] > max(
            result[c][group]["all_induced"] for c in ("previous", "permuted")
        )
    repairs = {}
    for group in before:
        matched = [
            (u, v)
            for u, v in zip(ledgers["previous"], ledgers["matched"], strict=True)
            if u["group"] == group
        ]
        repairs[group] = {
            metric: dict(
                gained=sum(v[metric] and not u[metric] for u, v in matched),
                lost=sum(u[metric] and not v[metric] for u, v in matched),
            )
            for metric in ("object", "induced", "both")
        }
    write_json(
        PUBLIC / "results.json",
        dict(
            arms=result,
            checks=checks,
            eligible=all(checks.values()),
            changes=repairs,
            raw_sha256={p.name: sha(p) for p in OUT.glob("*.npz")},
            ledger_sha256={p.name: sha(p) for p in OUT.glob("*_ledger.json")},
        ),
    )


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU0 required")
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(3)
        parser = argparse.ArgumentParser()
        parser.add_argument("--likelihood-only", action="store_true")
        evaluate(parser.parse_args().likelihood_only)
