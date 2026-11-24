"""Check word and stop decisions on gold histories against free-running physical replies."""

import gzip
import json
from collections import Counter
import torch
from scripts.answer_dialogue_ink import load
from scripts.dialogue_ink_data import Corpus, WordImages, read_data, pack
from scripts.prepare_dialogue_ink import WORK, RELEASE, PUB, dump, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


@torch.inference_mode()
def certify(model, decisions):
    net, opt = load(model, "cuda")
    data = read_data()
    held = json.loads(gzip.decompress(decisions.read_bytes()))
    face = Face(FACES[0])
    palette = net.palette.cpu().numpy()
    result = {}
    for group in ("reserved", "dev"):
        records = [r for r in held if r["group"] == group]
        ds = Corpus(data, group, net.reader, opt)
        words = [[face(w) for w in r["answer"].split()] for r in records]
        refbank = WordImages({w for ws in words for w in ws}, net.reader, opt)
        counts = {name: Counter() for name in ("all", "strict")}
        for start in range(0, len(records), 32):
            rows = records[start : start + 32]
            refs = words[start : start + 32]
            prev, _ = pack(
                [[0] + [refbank.ids[w] for w in ws[:16]] for ws in refs], "cuda"
            )
            x, variants = ds.inputs([r["index"] for r in rows])
            mem, _ = net.memory(*x)
            ch, ptr, z, _ = net.step(
                mem, refbank.features[prev.long(), :50], mem[-1][None]
            )
            route = ch.argmax(-1)
            ix = ptr.masked_fill(variants[:, None] == 0, -1e4).flatten(2).argmax(-1)
            copy = variants.flatten(1).gather(1, ix).tolist()
            loc = (route == 2).nonzero()
            spell = {}
            if loc.numel():
                ids, end = net.write_words(z[loc[:, 0], loc[:, 1]])
                for pos, v, ok in zip(
                    loc.tolist(), ids.tolist(), end.tolist(), strict=True
                ):
                    spell[tuple(pos)] = (
                        palette[[c for c in v if c]].tobytes() if ok else None
                    )
            for i, (r, ws, modes) in enumerate(
                zip(rows, refs, route.tolist(), strict=True)
            ):
                fails = []
                for t, w in enumerate(ws[:16]):
                    ch = modes[t]
                    actual = (
                        None
                        if ch == 0
                        else ds.bank.raw[copy[i][t]]
                        if ch == 1
                        else spell[i, t]
                    )
                    if actual != w:
                        fails.append(
                            ("early_stop", "wrong_pointer", "wrong_spelling")[ch]
                        )
                if len(ws) > 16:
                    fails.append("word_limit")
                elif modes[len(ws)] != 0:
                    fails.append("missing_stop")
                good = not fails
                for name in ["all", "strict"] if r["strict"] else ["all"]:
                    c = counts[name]
                    c["turns"] += 1
                    c["certified"] += good
                    c["free_exact"] += r["out"]["writer"]["pixels"]
                    c["disagreement"] += good != r["out"]["writer"]["pixels"]
                    if fails:
                        c["first_" + fails[0]] += 1
        result[group] = counts
    return dict(
        groups=result,
        decisions=sha(decisions),
        checkpoint=sha(model / "writer.safetensors"),
    )


if __name__ == "__main__":
    with gpu_owner():
        for name, model, run in [
            ("plain", RELEASE, "evaluation"),
            ("word_code", RELEASE / "word_code", "word_code_evaluation"),
        ]:
            result = certify(model, WORK / run / "decisions.json.gz")
            dump(PUB / name / "prefix_certificate.json", result)
            print(name, result, flush=True)
