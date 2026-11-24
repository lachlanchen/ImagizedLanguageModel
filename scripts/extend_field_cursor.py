"""Evaluate the frozen physical writer on additional natural CrossWOZ test turns."""

import gzip
import json
import zipfile
from collections import Counter
import torch
from scripts.prepare_chinese_two_fact import (
    RAW,
    PUB,
    WORK,
    SOURCE,
    MODEL,
    DOMAINS,
    FIELDS,
    WIDTH,
    read,
    sha,
    dump,
    value,
    Rasterizer,
)
from scripts.train_field_cursor import (
    convert_plans,
    load_cursor,
    labels,
    palette,
    cursor_actions,
)
from scripts.train_field_alignment import load_alignment
from scripts.evaluate_chinese_two_fact import (
    grade,
    surfaces,
    expand_text,
    keys_for,
)
from scripts.evaluate_field_cursor import attached
from scripts.train_semantic_page import gpu_owner


def collect():
    prior = {(r["dialog"], r["turn"]) for r in read(WORK / "test.json")}
    catalog = {d: dict(read(RAW / (f + "_db.json"))) for d, f in DOMAINS.items()}
    if (
        sha(RAW / "test.json.zip")
        != read(SOURCE / "corpus.json")["files"]["data/crosswoz/test.json.zip"]
    ):
        raise ValueError("Unexpected held source")
    with zipfile.ZipFile(RAW / "test.json.zip") as archive:
        dialogs = json.load(archive.open("test.json"))
    rows = []
    for did, dialog in dialogs.items():
        user = None
        for i, msg in enumerate(dialog["messages"]):
            if msg["role"] == "usr":
                user = msg
                continue
            if user is None or (did, i) in prior:
                continue
            request = [
                (d, k) for act, d, k, v in user["dialog_act"] if act == "Request"
            ]
            if len(request) < 2 or len(set(request)) != len(request):
                continue
            domain = request[0][0]
            if any(
                d != domain or d not in catalog or k not in FIELDS for d, k in request
            ):
                continue
            names = set(msg["sys_state"][domain].get("selectedResults", []))
            names.update(
                v
                for a, d, k, v in user["dialog_act"]
                if (a, d, k) == ("Inform", domain, "名称")
            )
            if len(names) != 1:
                continue
            name = next(iter(names))
            entry = catalog[domain].get(name, {})
            card = {k: value(k, entry.get(k)) for k in FIELDS}
            card = {k: v for k, v in card.items() if v and len(k) + 1 + len(v) <= WIDTH}
            keys = [k for _, k in request]
            q = user["content"]
            if (
                not 0 < len(q) <= WIDTH
                or len(name) + 3 > WIDTH
                or any(k not in card for k in keys)
            ):
                continue
            if any(list(card.values()).count(card[k]) != 1 for k in keys):
                continue
            if any(card[a] in card[b] for a in keys for b in keys if a != b):
                continue
            rows.append(
                dict(
                    dialog=did,
                    turn=i,
                    q=q,
                    domain=domain,
                    name=name,
                    card=card,
                    keys=keys,
                    answer=msg["content"],
                )
            )
    return rows


@torch.inference_mode()
def main():
    rows = collect()
    print("extension", Counter(len(r["keys"]) for r in rows), flush=True)
    dump(WORK / "extension.json", rows)
    chars = read(MODEL / "alphabet.json")
    raster = Rasterizer()
    trained = read(WORK / "train.json") + read(WORK / "joined.json")
    qs = {r["q"] for r in trained}
    names = {r["name"] for r in trained}
    results = {}
    with gpu_owner():
        writer = load_cursor("cuda")
        keys = writer.encode_labels(labels("cuda"))
        embedding = writer.glyph(palette("cuda"))
        for mode in ("ink", "byte", "human"):
            lines = []
            if mode != "human":
                planner = load_alignment(mode, "cuda")
                features = surfaces(rows, mode, raster)
            for start in range(0, len(rows), 48):
                part = rows[start : start + 48]
                if mode == "human":
                    predictions = [(r["answer"], []) for r in part]
                else:
                    dom = torch.tensor(
                        [list(DOMAINS).index(r["domain"]) for r in part], device="cuda"
                    )
                    avail = torch.tensor(
                        [[f in r["card"] for f in FIELDS] for r in part], device="cuda"
                    )
                    query = torch.as_tensor(
                        features[start : start + len(part)], device="cuda"
                    )
                    plan = planner.choose(query, dom, avail)
                    seq = writer.write(
                        keys,
                        dom + len(FIELDS),
                        convert_plans(plan, len(keys)),
                        embedding,
                    )
                    predictions = []
                    for r, p, s in zip(part, plan.tolist(), cursor_actions(seq, chars)):
                        fields = keys_for(p)
                        predictions.append((expand_text(s, chars, fields, r), fields))
                for r, (line, fields) in zip(part, predictions):
                    item = grade(line, fields, r)
                    item["attached"] = item["clean"] and attached(line, r["keys"], r)
                    lines.append(item)
            results[mode] = lines
    report = {}
    for mode, lines in results.items():
        by = {}
        for n in sorted({len(r["keys"]) for r in rows}):
            pairs = [(r, v) for r, v in zip(rows, lines) if len(r["keys"]) == n]
            by[n] = dict(
                n=len(pairs),
                clean=sum(v["clean"] for r, v in pairs),
                attached=sum(v["attached"] for r, v in pairs),
                novel_question=sum(r["q"] not in qs for r, v in pairs),
                novel_entity=sum(r["name"] not in names for r, v in pairs),
                both_new=sum(
                    r["q"] not in qs and r["name"] not in names for r, v in pairs
                ),
                both_new_clean=sum(
                    r["q"] not in qs and r["name"] not in names and v["clean"]
                    for r, v in pairs
                ),
            )
        report[mode] = by
    dump(
        PUB / "extension.json",
        dict(counts=report, dialogues=len({r["dialog"] for r in rows})),
    )
    (PUB / "extension-lines.json.gz").write_bytes(
        gzip.compress(
            json.dumps(dict(rows=rows, results=results), ensure_ascii=False).encode(),
            mtime=0,
        )
    )
    print(report)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
