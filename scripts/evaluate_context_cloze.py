"""Score real QA generation, conditional ranks and hidden words on new pages."""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from ilm.visual_lm.word_canvas import draw
from scripts.answer_dialogue_surface import load
from scripts.dialogue_ink_data import read_data
from scripts.dialogue_ink_images import WordImages
from scripts.evaluate_compose_dialogue import assignment, confidence, score, summarize
from scripts.evaluate_word_canvas import passes, write_bound
from scripts.evaluate_word_memory import assay
from scripts.measure_writer_energy import snapshot
from scripts.prepare_compose_dialogue import kind, prepared
from scripts.prepare_context_cloze import MODEL, PUB, reserved
from scripts.prepare_dialogue_ink import sha, dump
from scripts.prepare_word_memory import MODEL as PARENT
from scripts.train_context_cloze import page_state
from scripts.train_dialogue_subword import timed
from scripts.train_dialogue_surface import corpus, representation
from scripts.train_span_choice import gpu_owner


def rank(logits, truth):
    # Stable prototype order breaks ties, including all unseen fitting-prior words.
    target = logits.gather(1, truth[:, None])
    ids = torch.arange(logits.size(1), device=logits.device)[None]
    return 1 + ((logits > target) | ((logits == target) & (ids < truth[:, None]))).sum(
        1
    )


def ranks(net, ds, data, difficult, face, unit, prior, swaps):
    elig = [
        i for i in difficult if 1 <= len(data["rows"][i][5].split()) <= net.max_words
    ]
    raw = [[face(s) for s in data["rows"][i][5].split()] for i in elig]
    bank = WordImages(
        {w for line in raw for w in line}, net.reader, None, unit, net.encode_words
    )
    result = []
    for start in range(0, len(elig), 32):
        rows = np.asarray(elig[start : start + 32])
        words = raw[start : start + 32]
        lengths = torch.tensor([len(s) for s in words], device=ds.device)
        width = int(lengths.max())
        active = torch.arange(width, device=ds.device)[None] < lengths[:, None]
        ids = torch.zeros_like(active, dtype=torch.long)
        for i, s in enumerate(words):
            ids[i, : len(s)] = ids.new_tensor([bank.ids[w] for w in s])
        features = bank.features[ids, :50]
        memories = {}
        for key, q, p in (
            ("original", None, None),
            ("question", swaps[0][rows], None),
            ("passage", None, swaps[1][rows]),
        ):
            x, v = ds.inputs(rows, q, p)
            memories[key] = net.memory(*x)[0]
            if key == "original":
                available = [
                    {ds.bank.raw[j] for j in line if j}
                    for line in v.flatten(1).tolist()
                ]
        for t in range(width):
            bs = [
                b
                for b, line in enumerate(words)
                if t < len(line)
                and line[t] not in available[b]
                and line[t] in net.prototype_ids
            ]
            if not bs:
                continue
            truth = ids.new_tensor([net.prototype_ids[words[b][t]] for b in bs])
            visibility = active.clone()
            visibility[:, t] = False
            trials = {"prior": rank(prior.expand(len(bs), -1), truth).tolist()}
            for key in ("original", "question", "passage", "no_neighbours"):
                mask = (
                    torch.zeros_like(active) if key == "no_neighbours" else visibility
                )
                mem = memories.get(key, memories["original"])
                _, _, z = net.paint(mem, features, mask, lengths)
                logits = net.retrieve_logits(z[bs, t])
                trials[key] = rank(logits, truth).tolist()
            for i, b in enumerate(bs):
                result.append(
                    dict(
                        row=int(rows[b]), slot=t, **{k: v[i] for k, v in trials.items()}
                    )
                )
    return result


def page_words(net, ds, prior, out, tag):
    # Interior positions exclude the outer two words irrespective of target eligibility.
    pos = torch.arange(ds.page.size(1), device=ds.device)[None]
    coords = ((pos >= 2) & (pos + 2 < ds.pl[:, None])).nonzero()
    donor = coords.roll(len(coords) // 2, 0)
    if (donor[:, 0] == coords[:, 0]).any():
        raise ValueError("Exchanged neighbourhood includes same story")
    rows = []
    for start in range(0, len(coords), 256):
        c = coords[start : start + 256]
        z = page_state(net, ds, c)
        swapped = page_state(net, ds, donor[start : start + len(c)])
        left = z.clone()
        left[:, net.width + 2 * 52 :] = (
            0  # both future neighbours are hidden, within the line
        )
        predictions = [
            net.retrieve_logits(s).argmax(1).tolist() for s in (z, swapped, left)
        ]
        targets = net.lookup(ds.bank)[ds.page[c[:, 0], c[:, 1]].long()].tolist()
        for coord, truth, a, b, f in zip(
            c.tolist(), targets, *predictions, strict=True
        ):
            rows.append([*coord, truth, a, b, f])
    arr = np.array(rows)
    top = int(prior.argmax())
    return dict(
        n=len(rows),
        stored=int((arr[:, 2] >= 0).sum()),
        local=int((arr[:, 2] == arr[:, 3]).sum()),
        exchanged=int((arr[:, 2] == arr[:, 4]).sum()),
        left=int((arr[:, 2] == arr[:, 5]).sum()),
        unigram=int((arr[:, 2] == top).sum()),
        prior_mode=top,
        outputs=write_bound(out, tag + "-page", rows),
    )


def relative(a, b, rows):
    if [(r["row"], r["slot"]) for r in a] != [(r["row"], r["slot"]) for r in b]:
        raise ValueError("Different ranked positions")
    origin = np.array([1 / r["original"] for r in a])
    slots = [rows[r["row"]] for r in a]
    ix = np.arange(len(a))
    return {
        key: confidence(origin - np.array([1 / r[key] for r in records]), slots, ix)
        for key, records in (
            ("control", [{**r, "control": r["original"]} for r in b]),
            ("prior", a),
            ("question", a),
            ("passage", a),
            ("no_neighbours", a),
        )
    }


@torch.inference_mode()
def run(models, dest):
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
    out = dest / "responses"
    out.mkdir(parents=True, exist_ok=False)
    reg = read_data()
    extra = prepared()["extra"]
    pages = reg["pages"] | extra["pages"]
    pages_used = {r[1] for r in reg["rows"] + extra["rows"] if r[0] == "train"}
    counts = Counter(w for pid in sorted(pages_used) for w in pages[pid].split())
    study = dict(reservation=sha(PUB / "reservation.json"), models={}, comparisons={})
    for mode in ("ink", "byte"):
        face, unit, _ = representation(mode)
        decode = {face(chr(i)): chr(i) for i in range(32, 127)}

        def read(w):
            return "".join(decode[w[i : i + unit]] for i in range(0, len(w), unit))

        values, ranks_by_arm, grounded = {}, {}, {}
        for arm in ("parent", "qa", "passage"):
            clock, meter = monotonic(), snapshot()
            torch.cuda.reset_peak_memory_stats()
            folder = PARENT / mode if arm == "parent" else models / mode / arm
            net = load(folder, "cuda")
            ds = corpus(data, "fresh", net, face, unit)
            tag = mode + "-" + arm
            records = [{} for _ in rows]
            for condition, ix, x, v in passes(ds, subsets["all"], swaps, 32):
                for i, paths in zip(ix, draw(net, ds.bank, x, v), strict=True):
                    words, stop, _ = paths[0]
                    records[i][condition] = dict(
                        answer=" ".join(map(read, words)), stop=stop
                    )
            cost = timed(clock, meter)
            vals = score(records, refs, None, None)
            scores, grounded[arm] = summarize(vals, subsets)
            values[arm] = vals
            clock, meter = monotonic(), snapshot()
            hidden, slots = assay(
                net, ds, data, subsets["difficult"], face, unit, net.prototype_ids
            )
            meta = json.loads((models / mode / "qa/weights.json").read_text())
            prior = torch.tensor(meta["absent_counts"], device=ds.device)
            ranked = ranks(
                net, ds, data, subsets["difficult"], face, unit, prior, swaps
            )
            if len(ranked) != hidden["absent_stored"]["total"]:
                raise ValueError("Different stored-word denominator")
            ranks_by_arm[arm] = ranked
            summ = {}
            for key in ("original", "prior", "question", "passage", "no_neighbours"):
                rr = np.array([r[key] for r in ranked])
                summ[key] = dict(
                    n=len(rr),
                    top1=int((rr == 1).sum()),
                    top8=int((rr <= 8).sum()),
                    median=float(np.median(rr)),
                    mrr=float((1 / rr).mean()),
                )
            res = dict(
                weights=sha(folder / "writer.safetensors"),
                scores=scores,
                hidden=hidden,
                ranks=summ,
                evaluation=cost,
                diagnosis=timed(clock, meter),
                outputs=write_bound(out, tag, records),
                rank_outputs=write_bound(out, tag + "-ranks", ranked),
                slots=write_bound(out, tag + "-slots", slots),
            )
            if arm != "parent":
                clock, meter = monotonic(), snapshot()
                freq = torch.zeros_like(prior)
                for w, n in counts.items():
                    idx = net.prototype_ids.get(face(w))
                    if idx is not None:
                        freq[idx] += n
                res["pages"] = page_words(net, ds, freq, out, tag)
                res["page_evaluation"] = timed(clock, meter)
            study["models"][tag] = res
            print(
                tag,
                scores["difficult"]["exact"],
                summ["original"],
                res.get("pages", {}),
                flush=True,
            )
        for candidate, control in (
            ("qa", "parent"),
            ("passage", "parent"),
            ("passage", "qa"),
        ):
            d = values[candidate][:, 0, 1] - values[control][:, 0, 1]
            study["comparisons"][mode + "-" + candidate + "-" + control] = dict(
                ranks=relative(ranks_by_arm[candidate], ranks_by_arm[control], rows),
                answers={
                    k: dict(
                        f1=confidence(d, rows, ix),
                        gained=int(
                            (grounded[candidate][ix] & ~grounded[control][ix]).sum()
                        ),
                        lost=int(
                            (~grounded[candidate][ix] & grounded[control][ix]).sum()
                        ),
                    )
                    for k, ix in subsets.items()
                },
            )
    dump(dest / "results.json", study)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        run(args.models, args.output)
