"""Test the same property for two named venues, with the question and page fixed."""

import gzip
import json
import zipfile
from scripts.prepare_prose_fields import (
    BASE,
    PUBLIC,
    ROOT,
    LIMIT,
    FIELDS,
    DOMAINS,
    RAW,
    read,
    dump,
    value,
    norm,
)
from scripts.prose_entity_data import pairs, page
from scripts.evaluate_prose_fields import report, counts, save_png, transfer
from scripts.write_prose_fields import ProseReply, render, card_image, display
from scripts.prepare_chinese_card_reply import Rasterizer
from scripts.train_semantic_page import gpu_owner


def native_sources():
    catalogs = {d: dict(read(RAW / (f + "_db.json"))) for d, f in DOMAINS.items()}
    with zipfile.ZipFile(RAW / "test.json.zip") as z:
        ds = json.load(z.open("test.json"))
    seen = {norm(r["text"]) for r in read(BASE / "train-passages.json")}
    outputs = []
    for did, turns in ds.items():
        for i, m in enumerate(turns["messages"]):
            text = m["content"]
            if m["role"] != "sys" or len(text) > LIMIT or norm(text) in seen:
                continue
            for domain, db in catalogs.items():
                names = [n for n in db if n and text.count(n) == 1]
                names = [a for a in names if not any(a != b and a in b for b in names)]
                if len(names) != 2:
                    continue
                for k in FIELDS:
                    vals = [value(k, db[n].get(k)) for n in names]
                    if (
                        any(not v or text.count(v) != 1 for v in vals)
                        or vals[0] in vals[1]
                        or vals[1] in vals[0]
                    ):
                        continue
                    outputs.append((domain, *names, text, k, *vals))
    return outputs


def external_rows(sources):
    out = []
    for d, a, b, s, k, av, bv in sources:
        for n, v in ((a, av), (b, bv)):
            out.append(
                dict(
                    domain=d,
                    name=n,
                    text=s,
                    keys=[k],
                    q="请问这家的" + k + "是多少？",
                    answer="",
                    card={k: v},
                    spans={k: [s.index(v), s.index(v) + len(v)]},
                )
            )
    return out


def nearest(row, other):
    s = row["text"]
    lo = s.index(row["name"])
    hi = lo + len(row["name"])

    def dist(r):
        a, b = r["spans"][row["keys"][0]]
        return max(a - hi, lo - b, 0)

    return dist(row) < dist(other)


def main():
    groups = {}
    nn = {}
    seed = 14887
    for split in ("val", "test"):
        docs = read(BASE / (split + "-passages.json"))
        chosen = pairs(docs, seed, each=1)
        rows = [
            page(left, r, t, k)
            for a, b, k in chosen
            for left, r in ((a, b), (b, a))
            for t in (left, r)
        ]
        groups[split] = rows
        nn[split] = sum(
            all(nearest(rows[i + j], rows[i + (j ^ 1)]) for j in range(4))
            for i in range(0, len(rows), 4)
        )
    natural = native_sources()
    groups["native-multi"] = external_rows(natural)
    groups["authored-multi"] = external_rows(
        read(ROOT / "references/prose_entity_transfer.json")
    )
    groups["original-natural"] = read(BASE / "test-questions.json")["natural"]
    groups["original-missing"] = read(BASE / "test-questions.json")["missing"]
    groups["authored-single"] = transfer()
    raster = Rasterizer()
    scores = {}
    allresults = {}
    with gpu_owner(), zipfile.ZipFile(PUBLIC / "entity-examples.zip", "w") as z:
        for arm in ("base", "entity-erased", "entity-ink", "entity-byte"):
            mode = "byte" if arm.endswith("byte") else "ink"
            net = ProseReply("cuda", mode, arm=None if arm == "base" else arm)
            scores[arm] = {}
            allresults[arm] = {}
            for g, rows in groups.items():
                actual = []
                decode = dict(net.writer.characters)
                for c in set("".join(r["text"] + r["name"] for r in rows)):
                    decode.setdefault(raster.tile(c), c)
                for i in range(0, len(rows), 48):
                    rs = rows[i : i + 48]
                    pages = [
                        render(r["q"], r["domain"], r["name"], r["text"], raster)
                        for r in rs
                    ]
                    for j, (r, x, p) in enumerate(zip(rs, net.batch(pages, rs), pages)):
                        actual.append(report(r, x, net, decode))
                        if arm == "entity-ink" and (
                            (g == "test" and i + j < 20)
                            or g in ("native-multi", "authored-multi")
                        ):
                            k = f"{g}/{i+j:04}"
                            save_png(z, k + "-page.png", card_image(p))
                            if x["pixels"] is not None:
                                save_png(z, k + "-answer.png", display(x["pixels"]))
                score = counts(actual, rows)
                group_size = (
                    4 if g in ("val", "test") else 2 if g.endswith("-multi") else 0
                )
                if group_size:
                    passed = [
                        all(v["attached"] for v in actual[i : i + group_size])
                        for i in range(0, len(rows), group_size)
                    ]
                    score.update(
                        complete=sum(passed),
                        groups=len(passed),
                        per_field={
                            k: dict(
                                n=sum(r["keys"][0] == k for r in rows[::group_size]),
                                passed=sum(
                                    p and r["keys"][0] == k
                                    for p, r in zip(passed, rows[::group_size])
                                ),
                            )
                            for k in FIELDS
                        },
                    )
                scores[arm][g] = score
                allresults[arm][g] = actual
    dump(
        PUBLIC / "entity-scores.json",
        dict(results=scores, oracle_candidate_nearest=nn, native=len(natural)),
    )
    (PUBLIC / "entity-decisions.json.gz").write_bytes(
        gzip.compress(
            json.dumps(
                dict(rows=groups, results=allresults),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode(),
            mtime=0,
        )
    )
    print(
        {
            k: {
                g: (s.get("complete"), s.get("groups"), s["1"]["attached"])
                for g, s in gs.items()
            }
            for k, gs in scores.items()
        }
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
