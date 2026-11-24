"""Evaluate the complete continuous-prose writer, missing evidence and source edits."""

import gzip
import io
import json
import hashlib
import zipfile
from collections import Counter
from scripts.prepare_prose_fields import (
    BASE,
    PUBLIC,
    WEIGHTS,
    ROOT,
    FIELDS,
    DOMAINS,
    LIMIT,
    read,
    dump,
    sha,
    norm,
)
from scripts.train_prose_fields import train_rows
from scripts.train_semantic_page import gpu_owner
from scripts.write_prose_fields import ProseReply, render, card_image, display
from scripts.prepare_chinese_card_reply import Rasterizer
from scripts.evaluate_chinese_two_fact import grade
from scripts.evaluate_field_cursor import attached


def control_rows(passages):
    out = {"list": [], "delay": []}
    seen = set()
    for doc in passages:
        if norm(doc["text"]) in seen or len(doc["spans"]) < 2:
            continue
        seen.add(norm(doc["text"]))
        choices = [k for k in FIELDS if k in doc["spans"]][:2]
        card = {k: doc["text"][a:b] for k, (a, b) in doc["spans"].items()}
        for first, second in (choices, choices[::-1]):
            for k, query in (
                ("list", f"请按顺序介绍：{first}、{second}。"),
                ("delay", f"把{second}留到第二项，先回答{first}。"),
            ):
                out[k].append(
                    dict(doc, keys=[first, second], q=query, card=card, answer="")
                )
    return out


def transfer():
    rows = []
    for domain, name, text, a, b, f, g, q in read(
        ROOT / "references/prose_fields_transfer.json"
    ):
        keys = [f, g]
        card = {f: a, g: b}
        spans = {k: [text.index(v), text.index(v) + len(v)] for k, v in card.items()}
        rows.append(
            dict(
                name=name,
                text=text,
                domain=domain,
                keys=keys,
                q=q,
                card=card,
                spans=spans,
                answer="",
            )
        )
    return rows


def edits(rows, seen):
    out = []
    for i, r in enumerate(rows):
        k = r["keys"][0]
        v = r["card"][k]
        s = r["text"]
        if s.count(v) != 1 or v in r["q"]:
            continue
        a, b = r["spans"][k]
        for donor in rows[i + 1 :] + rows[:i]:
            w = donor["card"].get(k)
            if donor["name"] == r["name"] or not w or w == v or w in s:
                continue
            if len(s) - len(v) + len(w) > LIMIT:
                continue
            if norm(s[:a] + w + s[b:]) in seen:
                continue
            shift = len(w) - len(v)
            spans = {
                f: ([left + shift, h + shift] if left >= b else [left, h])
                for f, (left, h) in r["spans"].items()
            }
            if any(left < b and h > a for f, (left, h) in r["spans"].items() if f != k):
                continue
            spans[k] = [a, a + len(w)]
            out.append(
                dict(
                    r,
                    text=s[:a] + w + s[b:],
                    card=r["card"] | {k: w},
                    spans=spans,
                    parent=i,
                    old=v,
                    new=w,
                )
            )
            break
    return out


def report(r, item, model, decode, gold=False):
    fields = item["keys"]
    loc = r["spans"] if gold else item["intervals"]
    missing = [k for k in fields if k not in loc]
    value = {k: r["text"][a:b] for k, (a, b) in loc.items()}
    if gold and not missing:
        # Reuse the fitted actions and replace only their source bindings.
        marks = dict(zip(("◇", "◎", "△"), (r["name"], *[value[k] for k in fields])))
        line = "".join(
            marks.get(model.writer.alphabet[i], model.writer.alphabet[i])
            for i in item["actions"]
        )
    else:
        px = item["pixels"]
        line = (
            None
            if missing or px is None
            else "".join(decode.get(bytes(t), "\ufffc") for t in px)
        )
    ok = (
        line is not None
        and grade(line, fields, r)["clean"]
        and attached(line, r["keys"], r)
    )
    ordered = bool(
        ok and sorted(r["keys"], key=lambda k: line.index(r["card"][k])) == r["keys"]
    )
    return dict(
        keys=fields,
        intervals=loc,
        missing=missing,
        line=line,
        attached=bool(ok),
        ordered=ordered,
        plan=set(fields) == set(r["keys"]),
        ink=None
        if gold or item["pixels"] is None
        else hashlib.sha256(item["pixels"].tobytes()).hexdigest(),
    )


def counts(pred, rows):
    score = {}
    for n in (1, 2):
        pairs = [(x, r) for x, r in zip(pred, rows, strict=True) if len(r["keys"]) == n]
        score[str(n)] = dict(
            n=len(pairs),
            attached=sum(x["attached"] for x, r in pairs),
            ordered=sum(x["ordered"] for x, r in pairs),
            plan=sum(x["plan"] for x, r in pairs),
            abstain=sum(bool(x["missing"]) for x, r in pairs),
            right_refusal=sum(
                x["plan"]
                and set(x["missing"]) == set(r["keys"]) - set(r["spans"])
                and bool(x["missing"])
                for x, r in pairs
            ),
        )
    return score


def save_png(z, name, image):
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        z.writestr(info, buf.getvalue())


def evaluate():
    PUBLIC.mkdir(parents=True, exist_ok=True)
    raster = Rasterizer()
    corpus = read(BASE / "corpus.json")
    for n, h in corpus["sha256"].items():
        if sha(BASE / n) != h:
            raise ValueError(n)
    training, _ = train_rows()
    seen = {norm(r["text"]) for r in training}
    train_venues = {r["name"] for r in training}
    from scripts.prepare_chinese_card_reply import RAW, extract
    from scripts.prepare_chinese_two_fact import double_rows, paired_training

    catalog = {d: dict(read(RAW / (f + "_db.json"))) for d, f in DOMAINS.items()}
    singles, _ = extract("train", catalog)
    single_keys = [dict(r, keys=[r["key"]]) for r in singles]
    train_q = {
        r["q"]
        for rs in (
            single_keys,
            double_rows("train", catalog),
            paired_training(single_keys),
        )
        for r in rs
    }
    allrows = {}
    split_docs = {}
    for split in ("val", "test"):
        split_docs[split] = read(BASE / (split + "-passages.json"))
        for group, rows in read(BASE / (split + "-questions.json")).items():
            allrows[split + "-" + group] = rows
    allrows.update(control_rows(split_docs["test"]))
    allrows["authored"] = transfer()
    allrows["edits"] = edits(allrows["test-natural"], seen)
    for rs in [*split_docs.values(), *allrows.values()]:
        if any(norm(r["text"]) in seen for r in rs):
            raise ValueError("Evaluation passage used in a fit")
    results = {}
    scores = {}
    segment = {}
    schedule = {}
    folder = WEIGHTS / "examples"
    folder.mkdir(parents=True, exist_ok=True)
    with gpu_owner(), zipfile.ZipFile(PUBLIC / "test-images.zip", "w") as images:
        for mode in ("ink", "byte"):
            model = ProseReply("cuda", mode)
            dct = dict(model.writer.characters)
            # For scoring only, decode previously unseen source bitmaps from the supplied passage.
            for rows in allrows.values():
                for c in set("".join(r["text"] + r["name"] for r in rows)):
                    dct.setdefault(raster.tile(c), c)
            results[mode] = {}
            scores[mode] = {}
            segment[mode] = {}
            for split, docs in split_docs.items():
                total = Counter()
                fields = {k: Counter() for k in FIELDS}
                for start in range(0, len(docs), 48):
                    rs = docs[start : start + 48]
                    pages = [
                        render("？", r["domain"], r["name"], r["text"], raster)
                        for r in rs
                    ]
                    for r, x in zip(
                        rs, model.batch(pages, [r | {"q": "？"} for r in rs])
                    ):
                        for k in FIELDS:
                            e = r["spans"].get(k)
                            p = x["intervals"].get(k)
                            lab = (
                                "true_null"
                                if e is None and p is None
                                else "false_ink"
                                if e is None
                                else "miss"
                                if p is None
                                else "exact"
                                if r["text"][slice(*e)] == r["text"][slice(*p)]
                                else "boundary_or_role"
                            )
                            total[lab] += 1
                            fields[k][lab] += 1
                segment[mode][split] = dict(
                    total=total, fields=fields, passages=len(docs)
                )
            for group, rows in allrows.items():
                live = []
                oracle = []
                erased = []
                for start in range(0, len(rows), 48):
                    rs = rows[start : start + 48]
                    pages = [
                        render(r["q"], r["domain"], r["name"], r["text"], raster)
                        for r in rs
                    ]
                    items = model.batch(pages, rs)
                    blanks = model.batch(pages, rs, erase=True)
                    for j, (r, item, blank, page) in enumerate(
                        zip(rs, items, blanks, pages)
                    ):
                        pos = start + j
                        live.append(report(r, item, model, dct))
                        oracle.append(report(r, item, model, dct, gold=True))
                        erased.append(report(r, blank, model, dct))
                        if mode == "ink" and (
                            group.startswith("test-") or group in ("authored", "edits")
                        ):
                            stem = f"{group}/{pos:04}"
                            save_png(images, stem + "-page.png", card_image(page))
                            if item["pixels"] is not None:
                                save_png(
                                    images,
                                    stem + "-answer.png",
                                    display(item["pixels"]),
                                )
                results[mode][group] = dict(actual=live, oracle=oracle, erased=erased)
                gs = {k: counts(ls, rows) for k, ls in results[mode][group].items()}
                if group in ("list", "delay"):
                    gs["both_orders"] = sum(
                        left["ordered"] and r["ordered"]
                        for left, r in zip(live[::2], live[1::2])
                    )
                    gs["pairs"] = len(rows) // 2
                if group == "edits":
                    base = results[mode]["test-natural"]["actual"]
                    gs["both_edits"] = sum(
                        base[r["parent"]]["attached"] and p["attached"]
                        for r, p in zip(rows, live)
                    )
                scores[mode][group] = gs
                schedule[group] = dict(
                    n=len(rows),
                    unseen_questions=sum(r["q"] not in train_q for r in rows),
                    unseen_venues=sum(r["name"] not in train_venues for r in rows),
                )
    dump(PUBLIC / "corpus.json", corpus)
    output = json.dumps(
        dict(rows=allrows, results=results), ensure_ascii=False, separators=(",", ":")
    ).encode()
    (PUBLIC / "decisions.json.gz").write_bytes(gzip.compress(output, mtime=0))
    dump(
        PUBLIC / "scores.json",
        dict(
            groups=scores,
            segmentation=segment,
            overlap=schedule,
            images_sha256=sha(PUBLIC / "test-images.zip"),
            decisions_sha256=sha(PUBLIC / "decisions.json.gz"),
        ),
    )
    print(
        json.dumps(
            {g: v["actual"] for g, v in scores["ink"].items()}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    evaluate()
