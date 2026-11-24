"""Evaluate complete free replies, frozen single replies and their independent concatenation."""

import gzip
import json
import re
from itertools import combinations

import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_listener import beam_write
from scripts.prepare_chinese_two_fact import (
    WORK,
    PUB,
    MODEL,
    PARENT,
    DOMAINS,
    FIELDS,
    MARKS,
    WIDTH,
    Rasterizer,
    read,
    dump,
)
from scripts.train_chinese_two_fact import load, ARMS, archive, gpu_owner, output_bank
from scripts.train_chinese_card_reply import (
    load_writer,
    router,
    output_bank as parent_bank,
)
from scripts.write_chinese_two_fact import keys_for, write_plans
from scripts.write_chinese_card_reply import pack
from scripts.evaluate_chinese_card_reply import includes, CUES


def expand_text(seq, chars, keys, row):
    chunks = []
    refs = dict(zip(MARKS, [row["name"], *(row["card"][k] for k in keys)]))
    for i in seq:
        c = chars[i]
        chunks.append(refs[c] if c in MARKS else c)
    return "".join(chunks)


def grade(line, keys, row):
    wanted = row["keys"]
    targets = [row["card"][k] for k in wanted]
    coverage = all(includes(line, v) for v in targets)
    foreign = any(
        includes(line, v)
        for k, v in row["card"].items()
        if k not in wanted and not any(v in t for t in targets)
    )

    def nums(s):
        return set(re.findall("\\d+(?:\\.\\d+)?", s))

    extra = bool(nums(line) - nums(row["name"] + " " + " ".join(targets)))
    clean = coverage and not foreign and not extra
    actual = sorted(wanted, key=lambda k: line.find(row["card"][k])) if coverage else []
    explicit = all(row["q"].count(k) == 1 for k in wanted)
    order = sorted(wanted, key=row["q"].index) if explicit else []
    return dict(
        line=line,
        plan=keys,
        plan_set=set(keys) == set(wanted),
        both_values=coverage,
        clean=clean,
        answer_order=clean and actual == wanted,
        question_order=clean and actual == order if explicit else None,
        cues=clean and all(re.search(CUES[k], line) for k in wanted),
        other=foreign,
        other_number=extra,
        exact=line == row["answer"],
    )


def totals(rows):
    return {
        k: sum(bool(r[k]) for r in rows)
        for k in (
            "plan_set",
            "both_values",
            "clean",
            "answer_order",
            "question_order",
            "cues",
            "other",
            "other_number",
            "exact",
        )
    } | dict(
        n=len(rows), question_order_n=sum(r["question_order"] is not None for r in rows)
    )


def challenge(rows):
    seen = set()
    out = []
    for r in rows:
        for a, b in combinations(r["card"], 2):
            pair = r["domain"], a, b
            if pair in seen or r["card"][a] == r["card"][b]:
                continue
            seen.add(pair)
            for x, y in ((a, b), (b, a)):
                out.append(
                    dict(r, keys=[x, y], q=f"先说{x}，然后告诉我{y}。", answer="")
                )
    return out


def surfaces(rows, mode, raster):
    if mode == "ink":
        a = np.stack([pack(r["q"], raster) for r in rows])
        return observations({"tiles": a.reshape(-1, 128)}, "ink").reshape(
            len(rows), WIDTH, 256
        )
    buf = np.zeros((len(rows), WIDTH, 4), np.uint8)
    for i, r in enumerate(rows):
        for j, c in enumerate(r["q"]):
            buf[i, j, : len(c.encode())] = list(c.encode())
    return observations({"bytes": buf.reshape(-1, 4)}, "byte").reshape(
        len(rows), WIDTH, 256
    )


def independent(parent, keys, query, domains, available, plans, palette, rows, chars):
    result = [[] for _ in rows]
    for pos in range(2):
        index = [i for i, p in enumerate(plans) if p[pos] < len(FIELDS)]
        if not index:
            continue
        sel = torch.tensor([plans[i][pos] for i in index], device=query.device)
        _, mem = router(
            parent, query[index], keys, domains[index], available[index], sel
        )
        ids, lens = beam_write(
            parent,
            mem,
            torch.ones(mem.shape[:2], dtype=torch.bool, device=mem.device),
            palette,
            None,
            limit=120,
        )
        for i, s, n, c in zip(index, ids.tolist(), lens.tolist(), sel.tolist()):
            result[i].append(expand_text(s[:n], chars, [FIELDS[c]], rows[i]))
    return ["".join(pieces) for pieces in result]


@torch.inference_mode()
def main():
    meta = read(PUB / "corpus.json")
    bank = archive(WORK / "bank.npz", meta["bank"])
    chars = read(MODEL / "alphabet.json")
    held = {s: read(WORK / (s + ".json")) for s in ("val", "test")}
    held["swap"] = challenge(held["test"])
    ranked = {}
    scores = {}
    raster = Rasterizer()
    with gpu_owner():
        out = output_bank("cuda")
        base_out = parent_bank("cuda")
        base_chars = read(PARENT / "readback.json")
        parent = load_writer("ink", "cuda")
        labels = np.load(PARENT / "label-pixels.npy")
        base_labels = parent.encode_rows(
            torch.as_tensor(
                observations({"tiles": labels.reshape(-1, 128)}, "ink"), device="cuda"
            ).reshape(len(labels), WIDTH, 256)
        )
        for arm in (*ARMS, "ink-erased"):
            mode = arm.split("-")[0]
            net = load("ink-joined" if arm == "ink-erased" else arm, "cuda")
            cells = torch.as_tensor(observations(bank, mode), device="cuda")
            labels = net.encode_rows(cells[bank["labels"]])
            for split, records in held.items():
                inputs = surfaces(records, mode, raster)
                for start in range(0, len(records), 32):
                    rows = records[start : start + 32]
                    query = torch.as_tensor(
                        inputs[start : start + len(rows)], device="cuda"
                    )
                    if arm == "ink-erased":
                        query = torch.zeros_like(query)
                    domains = torch.tensor(
                        [list(DOMAINS).index(r["domain"]) for r in rows], device="cuda"
                    )
                    av = torch.tensor(
                        [[k in r["card"] for k in FIELDS] for r in rows], device="cuda"
                    )
                    plans, action = write_plans(
                        net, query, labels, domains, av, out, chars.index(MARKS[-1])
                    )

                    def save(key, lines, choices):
                        dest = ranked.setdefault(split, {}).setdefault(key, [])
                        dest.extend(
                            grade(line, keys_for(p), r)
                            for line, p, r in zip(lines, choices, rows)
                        )

                    save(
                        arm,
                        [
                            expand_text(s, chars, keys_for(p), r)
                            for s, p, r in zip(action, plans, rows)
                        ],
                        plans,
                    )
                    if arm == "ink-joined":
                        save(
                            "separate",
                            independent(
                                parent,
                                base_labels,
                                query,
                                domains,
                                av,
                                plans,
                                base_out,
                                rows,
                                base_chars,
                            ),
                            plans,
                        )
                        route, _ = router(parent, query, base_labels, domains, av)
                        single = [[i, len(FIELDS)] for i in route.argmax(1).tolist()]
                        save(
                            "single",
                            independent(
                                parent,
                                base_labels,
                                query,
                                domains,
                                av,
                                single,
                                base_out,
                                rows,
                                base_chars,
                            ),
                            single,
                        )
        for split, arms in ranked.items():
            records = held[split]
            scores[split] = {}
            for arm, pred in arms.items():
                groups = {}
                for count in (1, 2):
                    sub = [
                        r for r, src in zip(pred, records) if len(src["keys"]) == count
                    ]
                    if sub:
                        groups[str(count)] = totals(sub)
                if split == "swap":
                    groups["paired_order"] = sum(
                        a["question_order"] and b["question_order"]
                        for a, b in zip(pred[::2], pred[1::2])
                    )
                scores[split][arm] = groups
    dump(PUB / "results.json", scores)
    (PUB / "replies.json.gz").write_bytes(
        gzip.compress(json.dumps(ranked, ensure_ascii=False).encode(), mtime=0)
    )
    print(json.dumps(scores["test"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
