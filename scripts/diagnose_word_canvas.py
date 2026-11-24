"""Localize irreversible image errors using correct lengths and single hidden slots."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from scripts.answer_dialogue_surface import load
from scripts.train_dialogue_surface import representation, corpus
from scripts.measure_writer_energy import snapshot
from scripts.prepare_word_canvas import reserved, MODEL, PUB
from scripts.prepare_compose_dialogue import kind
from scripts.prepare_dialogue_ink import sha, dump
from scripts.evaluate_optical_passage_qa import metrics
from scripts.train_span_choice import gpu_owner
from scripts.train_dialogue_subword import timed
from scripts.evaluate_word_canvas import write_bound
from scripts.dialogue_ink_images import WordImages
from ilm.visual_lm.word_canvas import draw
from ilm.visual_lm.image_copy_search import copy_groups


@torch.inference_mode()
def diagnose(models, eval_dir, out):
    out.mkdir(parents=True, exist_ok=False)
    result = json.loads((eval_dir / "results.json").read_text())
    if result["reservation"] != sha(PUB / "reservation.json"):
        raise ValueError("Cohort mismatch")
    data = reserved()
    rows = data["rows"]
    eligible = [
        i
        for i, r in enumerate(rows)
        if kind(r[5], (data["pages"][r[1]], r[3], r[4])) != "ordinary"
        and 1 <= len(r[5].split()) <= 16
    ]
    final = {}
    for mode in ("ink", "byte"):
        binding = result["models"][mode + "-masked"]
        frozen = eval_dir / (mode + "-masked.json.gz")
        if (
            sha(frozen) != binding["outputs"]
            or sha(models / mode / "masked/writer.safetensors") != binding["weights"]
        ):
            raise ValueError("Frozen result mismatch")
        autonomous = json.loads(gzip.decompress(frozen.read_bytes()))
        clock = monotonic()
        meter = snapshot()
        torch.cuda.reset_peak_memory_stats()
        net = load(models / mode / "masked", "cuda")
        face, unit, _ = representation(mode)
        decode = {face(chr(i)): chr(i) for i in range(32, 127)}

        def read(w):
            return "".join(decode[w[i : i + unit]] for i in range(0, len(w), unit))

        ds = corpus(data, "fresh", net, face, unit)
        # Other target words are supplied as visible patches only in the single-slot assay.
        words = {face(w) for i in eligible for w in rows[i][5].split()}
        bank = WordImages(words, net.reader, None, unit, net.encode_words)
        sumcounts = Counter()
        records = []
        for start in range(0, len(eligible), 16):
            idx = eligible[start : start + 16]
            x, v = ds.inputs(np.array(idx))
            wanted = [[face(w) for w in rows[i][5].split()] for i in idx]
            lens = torch.tensor(list(map(len, wanted)), device=v.device)
            timeline = []
            pred = draw(
                net, ds.bank, x, v, supplied_lengths=lens, commit_rounds=timeline
            )
            memory, _ = net.memory(*x)
            w = int(lens.max())
            time = torch.arange(w, device=v.device)[None]
            active = time < lens[:, None]
            ids = torch.zeros_like(active, dtype=torch.long)
            for b, s in enumerate(wanted):
                ids[b, : len(s)] = torch.tensor(
                    [bank.ids[t] for t in s], device=v.device
                )
            seen = bank.features[ids, :50]
            ed = v.flatten(1).long()
            groups = copy_groups(v)
            table = torch.zeros_like(ed).scatter_(1, groups, ed)
            checks = [[] for _ in idx]
            for t in range(w):
                visible = active.clone()
                visible[:, t] = False
                route, ptr, z = net.paint(memory, seen, visible, lens)
                mass = ptr[:, t].flatten(1).masked_fill(ed == 0, -torch.inf).softmax(-1)
                agg = torch.zeros_like(mass).scatter_add_(1, groups, mass)
                chosen = table.gather(1, agg.argmax(1)[:, None])[:, 0].tolist()
                spelling = route[:, t, 1] > route[:, t, 0]
                glyphs, ended = net.write_words(z[:, t])
                letter = [bytes(c) for c in net.palette.cpu().numpy()]
                for b, s in enumerate(wanted):
                    if t >= len(s):
                        continue
                    copyable = ds.bank.ids.get(s[t], -1)
                    available = bool((ed[b] == copyable).any())
                    spelled = b"".join(letter[c] for c in glyphs[b].tolist() if c)
                    drawn = spelled if spelling[b] else ds.bank.raw[chosen[b]]
                    good = drawn == s[t] and (not bool(spelling[b]) or bool(ended[b]))
                    checks[b].append(
                        dict(
                            copyable=available,
                            spelled=bool(spelling[b]),
                            correct=bool(good),
                            pointer_correct=ds.bank.raw[chosen[b]] == s[t],
                            spell_correct=spelled == s[t] and bool(ended[b]),
                        )
                    )
            for b, i in enumerate(idx):
                words, stop, _ = pred[b][0]
                s = wanted[b]
                text = " ".join(map(read, words))
                first = min(
                    (
                        k
                        for k, g, a in zip(timeline[b], words, s, strict=True)
                        if g != a
                    ),
                    default=0,
                )
                score = metrics(text, [rows[i][5]])
                auto = autonomous[i]["original"]
                sumcounts.update(
                    turns=1,
                    free_literal=int(auto["stop"] and auto["answer"] == rows[i][5]),
                    free_exact=int(
                        auto["stop"] and metrics(auto["answer"], [rows[i][5]])[0]
                    ),
                    length_oracle_literal=int(stop and words == tuple(s)),
                    length_oracle_exact=int(stop and score[0]),
                    length_oracle_f1=score[1] if stop else 0,
                    fully_visible_context_all_correct=all(
                        x["correct"] for x in checks[b]
                    ),
                )
                sumcounts["first_error_pass_" + str(first)] += 1
                for c in checks[b]:
                    g = "copyable" if c["copyable"] else "absent"
                    sumcounts[g + "_words"] += 1
                    for k in ("correct", "pointer_correct", "spell_correct", "spelled"):
                        sumcounts[g + "_" + k] += c[k]
                records.append(
                    dict(
                        row=i,
                        oracle_length_answer=text,
                        stopped=stop,
                        commit_rounds=timeline[b],
                        first_literal_error_pass=first,
                        single_mask=checks[b],
                    )
                )
        cost = timed(clock, meter)
        final[mode] = dict(
            counts=dict(sumcounts),
            cost=cost,
            outputs=write_bound(out, mode + "-diagnosis", records),
        )
        print(mode, dict(sumcounts), flush=True)
    dump(out / "diagnosis.json", final)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=MODEL)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with gpu_owner():
        diagnose(a.models, a.evaluation, a.output)
