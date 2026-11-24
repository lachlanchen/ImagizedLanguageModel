"""Test autonomous answers, source exchanges and single concealed word images."""

import argparse
from pathlib import Path
from collections import Counter
from time import monotonic

import numpy as np
import torch
from ilm.visual_lm.image_copy_search import copy_groups
from ilm.visual_lm.word_canvas import draw
from scripts.answer_dialogue_surface import load
from scripts.dialogue_ink_images import WordImages
from scripts.evaluate_compose_dialogue import assignment, confidence, score, summarize
from scripts.evaluate_word_canvas import passes, write_bound
from scripts.measure_writer_energy import snapshot
from scripts.prepare_compose_dialogue import kind
from scripts.prepare_dialogue_ink import dump, sha
from scripts.prepare_word_canvas import MODEL as PARENTS
from scripts.prepare_word_memory import MODEL, PUB, reserved
from scripts.train_dialogue_subword import timed
from scripts.train_dialogue_surface import representation, corpus
from scripts.train_span_choice import gpu_owner


def assay(net, ds, data, ix, face, unit, prototypes):
    chosen = [i for i in ix if 1 <= len(data["rows"][i][5].split()) <= net.max_words]
    truths = [[face(w) for w in data["rows"][i][5].split()] for i in chosen]
    held = WordImages(
        {w for line in truths for w in line}, net.reader, None, unit, net.encode_words
    )
    palette = [bytes(c) for c in net.palette.cpu().numpy()]
    counts = {k: Counter() for k in ("copyable", "absent_stored", "absent_unstored")}
    records = []
    for start in range(0, len(chosen), 24):
        rows = chosen[start : start + 24]
        truth = truths[start : start + 24]
        inputs, variants = ds.inputs(np.array(rows))
        memory, _ = net.memory(*inputs)
        lengths = torch.tensor([len(r) for r in truth], device=variants.device)
        active = (
            torch.arange(int(lengths.max()), device=lengths.device)[None]
            < lengths[:, None]
        )
        ids = torch.zeros_like(active, dtype=torch.long)
        for b, line in enumerate(truth):
            ids[b, : len(line)] = ids.new_tensor([held.ids[w] for w in line])
        feats = held.features[ids, :50]
        ed = variants.flatten(1).long()
        groups = copy_groups(variants)
        table = torch.zeros_like(ed).scatter_(1, groups, ed)
        sources = [{ds.bank.raw[j] for j in row if j} for row in table.tolist()]
        for t in range(int(lengths.max())):
            visible = active.clone()
            visible[:, t] = False
            route, ptr, z = net.paint(memory, feats, visible, lengths)
            word, letters, ended, _, _ = net.choose_images(
                ds.bank,
                route[:, t : t + 1].log_softmax(-1),
                ptr[:, t : t + 1],
                z[:, t : t + 1],
                groups,
                table,
                ed != 0,
                active[:, t : t + 1],
            )
            for b, (j, glyph, stop) in enumerate(
                zip(
                    word[:, 0].tolist(),
                    letters[:, 0].tolist(),
                    ended[:, 0].tolist(),
                    strict=True,
                )
            ):
                if t >= len(truth[b]):
                    continue
                target = truth[b][t]
                out = (
                    net.prototype_words[-j - 1]
                    if j < 0
                    else ds.bank.raw[j]
                    if j
                    else b"".join(palette[c] for c in glyph if c)
                )
                group = (
                    "copyable"
                    if target in sources[b]
                    else "absent_stored"
                    if target in prototypes
                    else "absent_unstored"
                )
                good = stop and out == target
                counts[group].update(total=1, correct=int(good))
                records.append([int(rows[b]), t, group, bool(good)])
    return counts, records


@torch.inference_mode()
def evaluate(models, output):
    torch.set_num_threads(2)
    data = reserved()
    rows = data["rows"]
    refs = [[r[5]] for r in rows]
    cats = np.array([kind(r[5], (data["pages"][r[1]], r[3], r[4])) for r in rows])
    subsets = dict(
        all=np.arange(len(rows)),
        difficult=np.flatnonzero(cats != "ordinary"),
        assembly=np.flatnonzero(cats == "assembly"),
        spell=np.flatnonzero(cats == "spell"),
    )
    swaps = assignment(rows)
    out = output / "responses"
    out.mkdir(parents=True, exist_ok=False)
    study = dict(reservation=sha(PUB / "reservation.json"), models={}, differences={})
    for mode in ("ink", "byte"):
        face, unit, _ = representation(mode)
        decode = {face(chr(i)): chr(i) for i in range(32, 127)}

        def read(w):
            return "".join(decode[w[i : i + unit]] for i in range(0, len(w), unit))

        inventory = set(load(models / mode).prototype_words)
        values, grounded = {}, {}
        for arm in ("parent", "disabled", "memory"):
            clock, meter = monotonic(), snapshot()
            torch.cuda.reset_peak_memory_stats()
            folder = PARENTS / mode / "masked" if arm == "parent" else models / mode
            net = load(folder, "cuda")
            net.disable_retrieval = arm == "disabled"
            ds = corpus(data, "fresh", net, face, unit)
            records = [{} for _ in rows]
            for condition, ix, x, v in passes(ds, subsets["all"], swaps, 32):
                paths = draw(net, ds.bank, x, v)
                for i, trial in zip(ix, paths, strict=True):
                    words, stop, _ = trial[0]
                    records[i][condition] = dict(
                        answer=" ".join(map(read, words)), stop=stop
                    )
            cost = timed(clock, meter)
            vals = score(records, refs, None, None)
            scores, grounded[arm] = summarize(vals, subsets)
            values[arm] = vals
            clock, meter = monotonic(), snapshot()
            net.collect_certificates = True
            net.certificate_count = net.certificate_total = 0
            counts, slots = assay(
                net, ds, data, subsets["difficult"], face, unit, inventory
            )
            tag = mode + "-" + arm
            study["models"][tag] = dict(
                weights=sha(folder / "writer.safetensors"),
                scores=scores,
                hidden=counts,
                certified=net.certificate_count,
                tested_for_certificate=net.certificate_total,
                evaluation=cost,
                diagnosis=timed(clock, meter),
                outputs=write_bound(out, tag, records),
                slots=write_bound(out, tag + "-slots", slots),
            )
            print(tag, scores["difficult"]["exact"], counts, flush=True)
        for control in ("parent", "disabled"):
            dif = values["memory"][:, 0, 1] - values[control][:, 0, 1]
            for label, ix in subsets.items():
                study["differences"][f"{mode}-{control}-{label}"] = dict(
                    f1=confidence(dif, rows, ix),
                    gained=int((grounded["memory"][ix] & ~grounded[control][ix]).sum()),
                    lost=int((~grounded["memory"][ix] & grounded[control][ix]).sum()),
                )
    dump(output / "results.json", study)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gpu_owner():
        evaluate(args.models, args.output)
