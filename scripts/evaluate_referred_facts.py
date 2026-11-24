"""Resolve attribution and read the unchanged human property wording into a reply image."""

from collections import Counter
import gzip
import json
import torch
from scripts.prepare_prose_fields import BASE, LIMIT, read
from scripts.prepare_story_mentions import PUBLIC, MODEL, dump
from scripts.prose_entity_data import pairs
from scripts.write_referred_fact import ReferredFact
from scripts.write_masked_mentions import cast_image
from scripts.write_prose_fields import render, trim
from scripts.write_chinese_card_reply import card_image, display
from scripts.prepare_chinese_card_reply import Rasterizer
from scripts.evaluate_prose_fields import report


def main():
    torch.set_num_threads(2)
    raster = Rasterizer()
    net = ReferredFact()
    decode = dict(net.writer.writer.characters)
    c = Counter()
    saved = []
    docs = read(BASE / "test-passages.json")

    def evaluate(text, names, domain, key, values, question, example=False):
        for ch in set(text + "".join(names)):
            decode.setdefault(raster.tile(ch), ch)
        pages = [render(question, domain, n, text, raster) for n in names]
        panel = cast_image(names, raster)
        source = trim(pages[0][3:].reshape(-1, 128))
        mapping = net.bindings(source, panel)
        cropped = [net.restrict(p, mapping) for p in pages]
        results = {}
        for arm, batch in [
            ("direct", pages),
            ("referred", [item[0] for item in cropped]),
        ]:
            outputs = net.writer.batch(batch)
            answers = []
            for i, (v, r) in enumerate(zip(values, outputs)):
                info = dict(
                    text=text,
                    name=names[i],
                    domain=domain,
                    keys=[key],
                    q=question,
                    card={key: v},
                    spans={},
                    answer="",
                )
                item = report(info, r, net.writer, decode)
                if arm == "referred" and cropped[i][1]:
                    left, _ = cropped[i][1]
                    item["intervals"] = {
                        k: (a + left, b + left)
                        for k, (a, b) in item["intervals"].items()
                    }
                answers.append(item)
                if example:
                    dest = MODEL / "examples" / ("fee-" + str(i + 1))
                    dest.mkdir(parents=True, exist_ok=True)
                    card_image(pages[i]).save(dest / "input.png")
                    panel.save(dest / "cast.png")
                    if r["pixels"] is not None:
                        display(r["pixels"]).save(dest / (arm + "-answer.png"))
                        (dest / (arm + "-answer.bin")).write_bytes(
                            r["pixels"].tobytes()
                        )
            results[arm] = answers
        results["regions"] = [loc for p, loc in cropped]
        return results

    for a, b, key in pairs(docs, 14887, each=1):
        c["source_groups"] += 1
        rows = []
        for left, right in ((a, b), (b, a)):
            names = [left["name"], right["name"]]
            text = (
                names[0]
                + "和"
                + names[1]
                + "都有介绍。"
                + left["text"].replace(names[0], "前者")
                + "。"
                + right["text"].replace(names[1], "后者")
            )
            if len(text) > LIMIT:
                break
            values = [r["text"][slice(*r["spans"][key])] for r in (left, right)]
            rows.append(
                (text, names, left["domain"], key, values, f"请问这家的{key}是多少？")
            )
        if len(rows) != 2:
            c["overlength"] += 1
            continue
        result = [evaluate(*r) for r in rows]
        c["groups"] += 1
        for arm in ("direct", "referred"):
            good = [x["attached"] for v in result for x in v[arm]]
            c[arm + "_four"] += all(good)
            c[arm + "_answers"] += sum(good)
        c["routed"] += sum(x is not None for r in result for x in r["regions"])
        saved.append(result)
    park = evaluate(
        "清溪园和星河塔隔水相望。前者的门票是12元，后者收35元。",
        ["清溪园", "星河塔"],
        "景点",
        "门票",
        ["12元", "35元"],
        "请问门票是多少钱？",
        True,
    )
    dump(PUBLIC / "referred-facts.json", dict(summary=c, park=park))
    (PUBLIC / "referred-decisions.json.gz").write_bytes(
        gzip.compress(
            json.dumps(saved, ensure_ascii=False, separators=(",", ":")).encode(),
            mtime=0,
        )
    )
    print(
        c,
        [[v["line"] for v in park[arm]] for arm in ("direct", "referred")],
        flush=True,
    )


if __name__ == "__main__":
    main()
