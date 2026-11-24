"""Separate sentence expression from autonomous passage reading and change the supplied fact."""

from hashlib import sha256
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_visual_statements import WORK, PUB, MODEL, read, dump
from scripts.train_visual_statements import Statements
from scripts.train_paired_evidence import load_model, gpu_owner
from scripts.write_paired_evidence import reader, generate as read_fact
from scripts.write_recurrent_evidence import load
from scripts.write_visual_statements import editor, realize
from scripts.evaluate_recurrent_evidence import overlap
from scripts.evaluate_visual_cursor import normalize
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.evaluate_visual_discourse import draw, packed


@torch.inference_mode()
def main():
    data = Statements()
    held = read(WORK / "held.json")
    fresh = read(WORK / "fresh.json")
    chars = list(map(str, data.a["chars"]))
    tiles = list(map(bytes, data.a["tiles"]))
    book = read(PUB / "corpus.json")
    assert len(book["fresh"]) == 96
    face = Rasterizer()
    lookup = {v: i for i, v in enumerate(tiles) if i >= 2}

    def ids(t):
        seq = []
        for c in t:
            ink = face.tile(c)
            if ink not in lookup:
                lookup[ink] = len(tiles)
                tiles.append(ink)
                chars.append(c)
            seq.append(lookup[ink])
        return seq

    author = [
        (
            "school",
            "What year was the school founded?",
            "The library opened in 1640. The school was founded in 1635. Its first classroom was on the hill.",
            "1635",
        ),
        (
            "author",
            "Who wrote the account of the bronze inscriptions?",
            "Mira drew the map. Lena wrote the account of the bronze inscriptions. Oliver printed the plates.",
            "Lena",
        ),
        (
            "place",
            "Where was the fossil found?",
            "A fossil was found in Niger. The museum later displayed it in France.",
            "Niger",
        ),
        (
            "hours",
            "When does the evening class start?",
            "The morning class starts at eight. The evening class starts at six.",
            "six",
        ),
    ]
    authored = [
        dict(id=k, q=ids(q), s=ids(s), f=ids(f), source=s, question=q, refs=[f])
        for k, q, s, f in author
    ]
    modifications = []
    for i, r in enumerate(held):
        if (
            r["reference"].count(r["fact"]) != 1
            or len(r["fact"]) < 2
            or normalize(r["fact"]) in normalize(r["question"])
        ):
            continue
        for offset in range(1, len(held)):
            v = held[(i + offset) % len(held)]
            new = v["fact"]
            if (
                len(new) < 2
                or len(new) > 2 * len(r["fact"]) + 3
                or normalize(new) in normalize(r["reference"] + " " + r["question"])
            ):
                continue
            modifications.append(
                dict(
                    r,
                    f=v["f"],
                    fact=new,
                    reference=r["reference"].replace(r["fact"], new),
                    before=i,
                )
            )
            break
    # Apply a truly unencoded historical form as a changed evidence bitmap.
    plate = MODEL.parent / "paired_evidence/examples/yan/zh/oracle/reference.png"
    from ilm.visual_lm.portable_writer import read_image

    glyph = read_image(plate, 1)[0].tobytes()
    if glyph in lookup:
        raise ValueError("The historical mark is already in the training repertoire")
    gid = len(tiles)
    tiles.append(glyph)
    chars.append("�")
    tiles = np.stack([np.frombuffer(k, np.uint8) for k in tiles])
    x = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()

    def tx(seq):
        return "".join(chars[i] for i in seq)

    with gpu_owner():
        base = load_model("ink-pair", "cuda")
        retriever = reader("evidence", "cuda")

        def retrieve(rows):
            facts = []
            for k in range(0, len(rows), 16):
                g = rows[k : k + 16]
                q = [r["q"] for r in g]
                s = [r["s"] for r in g]
                facts += read_fact(
                    base, retriever, x, data.n, q, s, [a + b for a, b in zip(s, q)], 96
                )
            return facts

        rows = [r for r in held if r["s"]] + fresh + authored
        emitted = retrieve(rows)
        for r, (fact, stop) in zip(rows, emitted, strict=True):
            r.update(pred=fact, pred_stop=stop)
        for arm, net in (
            ("before", load("dynamic", "cuda")),
            ("trained", editor("cuda")),
        ):

            def outputs(rs, fs):
                result = []
                for k in range(0, len(rs), 16):
                    qs = [r["q"] for r in rs[k : k + 16]]
                    result += realize(
                        base, net, x, data.n, qs, fs[k : k + 16], data.space
                    )
                return result

            result = {}
            for name, group, field in (
                ("given", held, "f"),
                ("pipeline", [r for r in held if r["s"]], "pred"),
                ("fresh", fresh, "pred"),
                ("fresh_given", fresh, "f"),
                ("changed", modifications, "f"),
                ("public", authored, "pred"),
            ):
                out = outputs(group, [r[field] for r in group])
                details = []
                for r, (seq, stop) in zip(group, out, strict=True):
                    t = tx(seq)
                    refs = [r["fact"]] if "fact" in r else r["refs"]
                    accepts = lambda value: any(
                        normalize(value) == normalize(ref) for ref in refs
                    )
                    record = dict(
                        id=r["id"],
                        stopped=stop,
                        output=t,
                        shape_sha256=sha256(
                            tiles[np.asarray(seq, dtype=int)].tobytes()
                        ).hexdigest(),
                        emitted_fact=tx(r[field]),
                        fact_read=bool(
                            r.get("pred_stop", True)
                            and accepts(tx(r.get("pred", r["f"])))
                        ),
                        contains=bool(
                            stop
                            and any(
                                normalize(v) and normalize(v) in normalize(t)
                                for v in refs
                            )
                        ),
                    )
                    if "reference" in r:
                        record["exact"] = stop and normalize(
                            r["reference"]
                        ) == normalize(t)
                        record["lcs"] = overlap(t, r["reference"])
                    if name == "changed":
                        record["before"] = r["before"]
                    details.append(record)
                result[name] = details
                if arm == "trained" and name in (
                    "given",
                    "pipeline",
                    "fresh",
                    "public",
                ):
                    # Fixed order within each explicitly named stratum; public prompts always shown.
                    if name == "public":
                        choose = list(range(len(out)))
                    elif name == "fresh":
                        choose = list(range(4))
                    else:
                        choose = []
                        for hit in (True, False):
                            for i, v in enumerate(details):
                                if v.get("exact") == hit:
                                    choose.append(i)
                                    break
                    for i in choose:
                        r = group[i]
                        d = details[i]
                        seq = out[i][0]
                        dest = MODEL / "examples" / name / r["id"]
                        dest.mkdir(exist_ok=True, parents=True)
                        for tag, ink in (
                            ("question", r["q"]),
                            ("fact", r[field]),
                            ("answer", seq),
                            ("reference", r.get("y", r["f"])),
                        ):
                            draw(dest, tag, tiles[np.asarray(ink, dtype=int)])
                        if r.get("s"):
                            draw(dest, "passage", tiles[r["s"]])
                        dump(
                            dest / "readout.json",
                            dict(
                                d,
                                question=r["question"],
                                reference=r.get("reference"),
                                fact_reference=tx(r["f"]),
                                question_cells=len(r["q"]),
                                passage_cells=len(r.get("s", [])),
                                fact_cells=len(r[field]),
                                written=len(seq),
                            ),
                        )
            if arm == "trained":
                subset = modifications[:32]
                output = outputs(subset, [[gid] for _ in subset])
                result["historical_given"] = [
                    dict(
                        id=r["id"],
                        stopped=end,
                        physical_mark=gid in seq,
                        verbatim_only=seq == [gid],
                    )
                    for r, (seq, end) in zip(subset, output, strict=True)
                ]
            stats = {}
            for name, records in result.items():
                stats[name] = dict(n=len(records))
                for key in (
                    "exact",
                    "stopped",
                    "contains",
                    "fact_read",
                    "physical_mark",
                    "verbatim_only",
                ):
                    if key in records[0]:
                        stats[name][key] = sum(r[key] for r in records)
                if "lcs" in records[0]:
                    stats[name]["mean_lcs"] = float(
                        np.mean([r["lcs"] for r in records])
                    )
            stats["changed"]["both_exact"] = sum(
                r["exact"] and result["given"][r["before"]]["exact"]
                for r in result["changed"]
            )
            stats["fresh"]["correct_fact_preserved"] = sum(
                r["fact_read"] and r["contains"] for r in result["fresh"]
            )
            packed(PUB / (arm + "-answers.json.gz"), result)
            dump(PUB / (arm + "-scores.json"), stats)
            print(arm, stats, flush=True)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
