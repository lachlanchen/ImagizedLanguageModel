"""Score actual drawn words, paired same-head senses and pictured-question retention."""

from collections import Counter, defaultdict
import hashlib
import numpy as np
import torch

from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.gloss_focus import cell_features
from scripts.evaluate_everyday_gloss import DEST, score
from scripts.prepare_asked_gloss import dump, read, sha
from scripts.prepare_pronoun_writer import PUBLIC, WORK, window
from scripts.train_asked_gloss import packed_batch
from scripts.write_visual_gloss import load_adapter
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore

NAMES = ["parent", "aligned", "exchanged"]


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    spec = read(PUBLIC / "split.json")
    assert sha(PUBLIC / "held.json") == spec["test_sha256"]
    held = read(PUBLIC / "held.json")
    face = Face(FACES[0])
    table = {face.cell(c): c for c in "abcdefghijklmnopqrstuvwxyz-"}
    writer = restore()
    readers = {"parent": None, **{a: load_adapter(WORK / a, "cpu") for a in NAMES[1:]}}
    classes, records = {}, []

    def draw(batch, adapter):
        src, lens = packed_batch(batch, face, "cpu")
        assert (lens == 17).all()
        codes, ends = writer.write(cell_features(src), context_adapter=adapter)
        raw = np.packbits(writer.draw(codes).numpy(), axis=-1)
        results = []
        for c, b, e in zip(codes.numpy(), raw, ends.tolist()):
            b = b[c != writer.stop]
            results.append(
                (
                    "".join(table.get(t.tobytes(), "�") for t in b)
                    if e
                    else "<unfinished>",
                    b,
                )
            )
        return results

    examples = PUBLIC / "samples"
    examples.mkdir(exist_ok=True)
    for a in range(0, len(held), 48):
        rows = held[a : a + 48]
        outputs = {
            name: draw([r[0] for r in rows], net) for name, net in readers.items()
        }
        for j, (w, ref, kind, book, sid, part) in enumerate(rows):
            words = {name: out[j][0] for name, out in outputs.items()}
            rec = dict(
                window=w,
                reference=ref,
                kind=kind,
                book=book,
                sid=sid,
                part=part,
                words=words,
            )
            key = (part, kind, "correct" if words["aligned"] == ref else "miss")
            if key not in classes and w[8] == "之":
                tag = "-".join(key)
                classes[key] = a + j
                write_strip(
                    np.frombuffer(face(w), np.uint8).reshape(-1, 128),
                    examples / (tag + ".png"),
                )
                for name, out in outputs.items():
                    write_strip(out[j][1], examples / (tag + "-" + name + ".png"))
                rec["image"] = tag
            records.append(rec)
    totals = defaultdict(Counter)
    pairs = {}
    for part in ["held_book", "held_section"]:
        groups = defaultdict(list)
        for j, r in enumerate(records):
            if r["part"] != part:
                continue
            group = totals[part + "/" + r["kind"]]
            group["count"] += 1
            for name in NAMES:
                group[name] += r["words"][name] == r["reference"]
            if r["window"][8] == "之":
                groups[r["kind"]].append(j)
        for ids in groups.values():
            ids.sort(
                key=lambda j: hashlib.sha256(records[j]["window"].encode()).hexdigest()
            )
        matched = list(zip(groups["motion"], groups["object_person"]))
        pairs[part] = dict(
            count=len(matched),
            indices=matched,
            correct={
                n: sum(
                    all(records[i]["words"][n] == records[i]["reference"] for i in pair)
                    for pair in matched
                )
                for n in NAMES
            },
        )
    source = read(DEST / "positions.json")
    # The reader and all chosen source locations stay fixed when the writer changes.
    strings = sorted(
        {window(r["page"], i) for r in source for i in range(len(r["page"]))}
    )
    all_outputs = defaultdict(dict)
    for a in range(0, len(strings), 64):
        batch = strings[a : a + 64]
        for name, net in readers.items():
            all_outputs[name].update(
                (k, v[0]) for k, v in zip(batch, draw(batch, net), strict=True)
            )
    transfers = {}
    for name in NAMES:
        revised = [
            dict(
                r,
                words=[
                    all_outputs[name][window(r["page"], i)]
                    for i in range(len(r["page"]))
                ],
            )
            for r in source
        ]
        if name == "parent":
            assert all(
                r["words"] == o["words"] for r, o in zip(revised, source, strict=True)
            )
        transfers[name] = score(revised)["everyday"]
    dump(PUBLIC / "held_replies.json", records)
    dump(PUBLIC / "transfer_words.json", all_outputs)
    dump(
        PUBLIC / "result.json",
        dict(
            totals=totals,
            pairs=pairs,
            transfer=transfers,
            held_sha256=sha(PUBLIC / "held_replies.json"),
            transfer_sha256=sha(PUBLIC / "transfer_words.json"),
        ),
    )
    print(dict(totals), {p: v["correct"] for p, v in pairs.items()})
    for name in NAMES:
        print(
            name,
            {
                g: (v["classroom"], v["writable"])
                for g, v in transfers[name].items()
                if g in ("fresh", "previous")
            },
        )


if __name__ == "__main__":
    main()
