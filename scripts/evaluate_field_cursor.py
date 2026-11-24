"""Score the scratch-trained field realizer and check local word-to-value attachments."""

import gzip
import json
import re
import torch
from scripts.prepare_chinese_two_fact import (
    ROOT,
    MODEL,
    WORK,
    PUB,
    DOMAINS,
    FIELDS,
    read,
    dump,
    Rasterizer,
)
from scripts.train_field_cursor import (
    load_cursor,
    labels,
    palette,
    convert_plans,
    cursor_actions,
)
from scripts.train_field_alignment import load_alignment
from scripts.train_semantic_page import gpu_owner
from scripts.evaluate_chinese_two_fact import (
    grade,
    totals,
    expand_text,
    keys_for,
    surfaces,
    challenge,
)
from scripts.evaluate_chinese_card_reply import CUES


def attached(line, fields, row):
    values = row["card"]
    if any(line.count(values[k]) != 1 for k in fields):
        return False
    masked = line
    for i, key in enumerate(fields):
        masked = masked.replace(values[key], chr(0xE000 + i))
    clauses = re.split("[。！？；，,;]", masked)
    for i, key in enumerate(fields):
        marker = chr(0xE000 + i)
        part = next(c for c in clauses if marker in c)
        left, right = part.split(marker)
        # Never cross another bound value when looking for its governing cue.
        left = re.split("[\ue000-\ue008]", left)[-1]
        right = re.split("[\ue000-\ue008]", right)[0]
        cues = []
        for k, pattern in CUES.items():
            if k == "酒店类型":
                pattern += "|酒店$"
            for m in re.finditer(pattern, left):
                cues.append((len(left) - m.end(), k))
        if not cues:
            for k, pattern in CUES.items():
                if k == "酒店类型":
                    pattern += "|酒店"
                cues.extend((m.start(), k) for m in re.finditer(pattern, right))
        if not cues:
            return False
        possible = {k for dist, k in cues if dist == min(n for n, k in cues)}
        if key not in possible and not (
            key in ("人均消费", "门票") and "价格" in possible
        ):
            return False
    return True


@torch.inference_mode()
def main():
    held = {s: read(WORK / (s + ".json")) for s in ("val", "test")}
    held["swap"] = challenge(held["test"])
    for name, frame in (
        ("fresh-list", "请按顺序介绍：{a}、{b}。"),
        ("fresh-delay", "把{b}留到第二项，先回答{a}。"),
    ):
        held[name] = [
            dict(r, q=frame.format(a=r["keys"][0], b=r["keys"][1]))
            for r in held["swap"]
            if len(r["card"]) > 2
        ]
    scores = {}
    raster = Rasterizer()
    chars = read(MODEL / "alphabet.json")
    with gpu_owner():
        writer = load_cursor("cuda")
        keys = writer.encode_labels(labels("cuda"))
        embed = writer.glyph(palette("cuda"))
        for mode in ("ink", "byte"):
            planner = load_alignment(mode, "cuda")
            scores[mode] = {}
            for split, rows in held.items():
                inputs = surfaces(rows, mode, raster)
                lines = []
                for start in range(0, len(rows), 48):
                    part = rows[start : start + 48]
                    q = torch.as_tensor(
                        inputs[start : start + len(part)], device="cuda"
                    )
                    dom = torch.tensor(
                        [list(DOMAINS).index(r["domain"]) for r in part], device="cuda"
                    )
                    av = torch.tensor(
                        [[k in r["card"] for k in FIELDS] for r in part], device="cuda"
                    )
                    plan = planner.choose(q, dom, av)
                    seq = writer.write(
                        keys, dom + len(FIELDS), convert_plans(plan, len(keys)), embed
                    )
                    for r, p, s in zip(part, plan.tolist(), cursor_actions(seq, chars)):
                        fields = keys_for(p)
                        lines.append(grade(expand_text(s, chars, fields, r), fields, r))
                groups = {
                    str(n): totals(
                        [s for s, r in zip(lines, rows) if len(r["keys"]) == n]
                    )
                    for n in (1, 2)
                }
                scores[mode][split] = dict(groups=groups, lines=lines)
                if split not in ("val", "test"):
                    scores[mode][split]["paired_order"] = sum(
                        a["answer_order"] and b["answer_order"]
                        for a, b in zip(lines[::2], lines[1::2])
                    )
    dump(PUB / "cursor-results.json", scores)
    records = [r for r in held["test"] if len(r["keys"]) == 2]
    n = len(records)
    controls = ROOT / "references/evidence/chinese_two_fact_v1"
    preds = json.loads(gzip.decompress((controls / "replies.json.gz").read_bytes()))[
        "test"
    ]
    arms = {k: [x["line"] for x in v[-n:]] for k, v in preds.items()}
    arms.update(
        {
            k: [x["line"] for x in read(controls / (f + ".json"))["test"]["lines"]]
            for k, f in (
                ("covered", "coverage"),
                ("aligned", "alignment-ink"),
                ("aligned-byte", "alignment-byte"),
            )
        }
    )
    arms["aligned-separate"] = [
        x["line"] for x in read(controls / "aligned-separate.json")["lines"]
    ]
    arms["human"] = [r["answer"] for r in records]
    arms["qwen"] = [s["text"] for s in read(controls / "qwen-answer.json")["outputs"]]
    for mode in ("ink", "byte"):
        arms["cursor-" + mode] = [q["line"] for q in scores[mode]["test"]["lines"][-n:]]
    checks = {}
    for arm, ls in arms.items():
        checks[arm] = [
            bool(grade(s, [], r)["clean"] and attached(s, r["keys"], r))
            for s, r in zip(ls, records, strict=True)
        ]
    dump(
        PUB / "attachment.json",
        dict(passed={k: sum(v) for k, v in checks.items()}, per_item=checks),
    )
    print({k: sum(v) for k, v in checks.items()})


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
