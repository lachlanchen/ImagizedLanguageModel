"""Factor visible evidence, reference fact and sibling query on the same natural English pages."""

from collections import defaultdict
from hashlib import sha256
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_evidence_views import Alphabet
from scripts.prepare_region_writer import Ink, MODEL as BASE
from scripts.prepare_marginal_confirmation import WORK, PUB
from scripts.train_marginal_phrase import MODEL
from scripts.prepare_open_exam import read, dump
from scripts.write_region_writer import model, phrase_model
from scripts.write_visual_discourse import Writer
from scripts.write_visual_statements import editor, fact_page
from scripts.write_recurrent_evidence import generate
from scripts.evaluate_region_writer import run_group, changed, contains_fact
from scripts.audit_region_writer import wording
from scripts.train_paired_evidence import gpu_owner
from scripts.evaluate_visual_discourse import packed, unpack


def main():
    rows = [r for r in read(WORK / "confirmation.json") if r["task"] == "statements"]
    edited = changed(rows)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["family"]].append(i)
    swap = {i: (v[0] if i != v[0] else v[1]) for v in groups.values() for i in v}
    assert all(len(v) == 2 for v in groups.values())

    def substitute(rs):
        return [{**r, "blocks": [r["facts"]["answer"]], "valid": [0]} for r in rs]

    facts = substitute(rows)
    moved = [{**r, "q": rows[swap[i]]["q"]} for i, r in enumerate(facts)]
    data = {
        "human_region": rows,
        "human_fact": facts,
        "sibling_and_fact": moved,
        "edited_fact": substitute(edited),
    }

    def counts(rs, z, ink):
        exact = []
        hits = []
        for r, v in zip(rs, z, strict=True):
            raw = bytes.fromhex(v["ink_hex"])
            assert sha256(raw).hexdigest() == v["sha256"]
            exact.append(
                v["stop"]
                and any(
                    b"".join(ink.tiles[k] for k in ink.encode(a)) == raw
                    for a in r["answers"]
                )
            )
            hits.append(
                v["stop"]
                and contains_fact(v["text"], r["facts"]["answer"], "statements")
            )
        return dict(n=len(z), exact=sum(exact), fact=sum(hits)), exact

    summary = {}
    report = {}
    with gpu_owner(), torch.inference_mode():
        torch.set_float32_matmul_precision("highest")
        sel = model("selector", "cuda")
        for a, net in (
            ("single", model("selected", "cuda")),
            ("marginal", phrase_model(MODEL, "cuda")),
        ):
            ink = Ink(np.load(BASE / "palette.npy"))
            book = read(BASE / "readback.json")
            z = {
                k: run_group(net, sel, "oracle", v, ink, book, None, wording)
                for k, v in data.items()
            }
            z["ordinary"] = [
                v
                for v in unpack(WORK / ("new-" + a + ".json.gz"))["original"]
                if v["task"] == "statements"
            ]
            report[a] = z
        alphabet = Alphabet()
        space = alphabet.encode(" ")[0]
        pairs = {
            k: [
                (alphabet.encode(r["q"]), alphabet.encode(r["facts"]["answer"]))
                for r in data[k]
            ]
            for k in ("human_fact", "sibling_and_fact", "edited_fact")
        }
        tiles = alphabet.array()
        features = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
        writer = Writer(device="cuda")
        ed = editor("cuda")
        report["specialist"] = {}
        for k, inputs in pairs.items():
            outputs = []
            for j in range(0, len(inputs), 12):
                batch = inputs[j : j + 12]
                q = [pair[0] for pair in batch]
                p = [fact_page(qi, fi, space) for qi, fi in batch]
                for ids, end in generate(
                    writer.model,
                    ed,
                    features,
                    alphabet.n,
                    q,
                    p,
                    p,
                    256,
                    geometry=(320, 4096),
                ):
                    raw = tiles[np.asarray(ids, int)].tobytes()
                    outputs.append(
                        dict(
                            text="".join(alphabet.chars[i] for i in ids),
                            stop=end,
                            ink_hex=raw.hex(),
                            sha256=sha256(raw).hexdigest(),
                        )
                    )
            report["specialist"][k] = outputs
    ink = Ink(np.load(BASE / "palette.npy"))
    for a, d in report.items():
        summary[a] = {}
        exact = {}
        for k, z in d.items():
            rs = edited if k == "edited_fact" else rows
            summary[a][k], exact[k] = counts(rs, z, ink)
        summary[a]["edits_both_exact"] = sum(
            exact["human_fact"][r["original"]] and b
            for r, b in zip(edited, exact["edited_fact"], strict=True)
        )
    packed(WORK / "fact-interventions.json.gz", report)
    dump(PUB / "fact-interventions.json", summary)
    print(summary)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
