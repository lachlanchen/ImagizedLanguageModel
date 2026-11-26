"""Check CPU deployment against archived print, dictionary and bilingual outputs."""

import json
from zipfile import ZipFile
import numpy as np
import pyarrow.parquet as pq
import torch

from ilm.visual_lm.open_source_cells import OpenSourceCells
from ilm.visual_lm.passage_selector import PassageWriter
from scripts.evaluate_book_order_prior import ANSWERS, png, unpack, restore
from scripts.evaluate_origin_value_reader import ARCHIVE as ORIGINS
from scripts.evaluate_book_page_memory import RAW, normalize
from scripts.evaluate_passage_selector import ARCHIVE, read_array, read_ink
from scripts.prepare_passage_selector import WORK, PUB, base
from scripts.write_origin_values import answer, decode, model


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    receipts = {}
    with ZipFile(ANSWERS) as z:
        rs = json.loads(z.read("records.json"))
        w = restore("intact", "adapted", "cpu")
        routes = [OpenSourceCells(w, t) for t in (0, 1)]
        selected = list(
            dict.fromkeys(
                i
                for case in (
                    lambda r: r["group"] == "news",
                    lambda r: r.get("open"),
                    lambda r: r["direction"] == 1,
                )
                for i in [j for j, r in enumerate(rs) if case(r)][:8]
            )
        )
        same = []
        for i in selected:
            r = rs[i]
            raw = unpack(z.read(f"{i}/source.png"), r["input"]["cells"])
            pixels, stop = routes[r["direction"]].translate(raw)
            same.append(
                bool(
                    stop == r["answers"]["intact"]["stop"]
                    and png(pixels) == z.read(f"{i}/intact.png")
                )
            )
        receipts["bilingual"] = dict(indices=selected, same=same)
    with ZipFile(ORIGINS) as z:
        rows = json.loads(z.read("records.json"))
        net = model("matched", "cpu")
        selected = [
            r
            for kind in ("original", "values", "headword")
            for r in [x for x in rows if x["kind"] == kind][:8]
        ]
        items = [f"{r['i']:04d}/{r['kind']}" for r in selected]
        sources = [
            decode(z.read(p + "/question.png"), z.read(p + "/page.png"), r["geometry"])
            for r, p in zip(selected, items, strict=True)
        ]
        outputs = answer(net, sources, "cpu")
        receipts["etymology"] = dict(
            indices=items,
            same=[
                bool(
                    blob == z.read(p + "/matched.png")
                    and count == r["models"]["matched"]["cells"]
                    and stop == r["models"]["matched"]["stop"]
                )
                for (blob, count, stop), p, r in zip(
                    outputs, items, selected, strict=True
                )
            ],
        )
    face = base.Ink()
    table = pq.read_table(RAW / "lambada-validation.parquet", columns=["text"])[
        "text"
    ].to_pylist()
    forms = base.read(WORK / "forms.json")
    separators = [" ", ". ", '" ', "; ", ", ", ": ", "- ", "( "]
    with ZipFile(ARCHIVE) as z:
        for arm in ("intact", "shuffled"):
            writer = PassageWriter(arm)
            checked = []
            for split in ("fiction", "gutenberg", "lambada"):
                with np.load(WORK / (split + ".npz")) as a:
                    src, marks, task, group = [
                        a[k] for k in ("source", "marks", "task", "group")
                    ]
                raw = read_ink(z, split + "/" + arm)
                flags = read_array(z, split + "/" + arm + "-stop")
                sample = np.linspace(0, len(src) - 1, 16).astype(int)
                for i in sample:
                    if split == "lambada":
                        s = normalize(table[group[i]]).rsplit(" ", 1)[0] + " "
                    else:
                        s = "".join(
                            forms[w] + separators[m]
                            for w, m in zip(src[i], marks[i], strict=True)
                            if w
                        )
                    ids = face.encode(s)
                    image = face.array()[ids]
                    words, gaps = writer.face.scan(image)
                    identical_input = words == [
                        writer.face.ids(forms[x]) for x in src[i] if x
                    ] and gaps == list(marks[i][src[i] != 0])
                    result, stop = writer.write(image, bool(task[i]), 1)
                    checked.append(
                        dict(
                            split=split,
                            row=int(i),
                            input=bool(identical_input),
                            pixels=bool(np.array_equal(result, raw[i])),
                            stop=bool(stop == flags[i]),
                        )
                    )
            receipts[arm] = checked
    base.dump(PUB / "replay.json", dict(archive=base.sha(ARCHIVE), results=receipts))
    for k, v in receipts.items():
        if isinstance(v, dict):
            print(k, sum(v["same"]), len(v["same"]))
        else:
            print(
                k,
                {
                    field: sum(r[field] for r in v)
                    for field in ("input", "pixels", "stop")
                },
                len(v),
            )


if __name__ == "__main__":
    main()
