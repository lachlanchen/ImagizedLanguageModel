"""Test automatic pictured record selection with the frozen shared passage and entry writer."""
from collections import Counter
import json

import numpy as np
import torch

from ilm.visual_lm.pictured_record_selection import select_visible_record
from scripts.audit_general_entry_writer import EVIDENCE, archived
from scripts.evaluate_general_entry_writer import english_request, pictured
from scripts.evaluate_same_passage_writer import generate, image_archive
from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.train_general_entry_writer import (AUTHORED, DEVICE, Entries, PairCorpus, Rasterizer,
                    WORK, load_writer, memory_bank, physical_tables, small)


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    bank = memory_bank()
    ds = Entries(bank, "development")
    dr = PairCorpus(bank, "development")
    du, _ = small(bank, "development")
    r = Rasterizer()
    english = [pictured(bank, r, english_request(m)) for m in ds.meta]
    authored = [json.loads(l) for l in AUTHORED.read_text().splitlines()]
    free = [pictured(bank, r, m["text"]) for m in authored]
    modes = (("simple", ds.q, ds.rows), ("paraphrase", ds.paraphrases, ds.rows),
             ("independent", free, [m["row"] for m in authored]), ("english", english, ds.rows),
             ("erased", [np.array([bank.lookup[bytes(128)]])]*len(ds.rows), ds.rows))
    keys = [bank.lookup[r.tile(c)] for c in "【】"]
    features, mem, _ = physical_tables(bank, DEVICE)
    net = load_writer(WORK / "paired", DEVICE)
    ink = net.ink(features)
    writing = net.write_key(ink[mem])
    root = WORK / "visible_selection"
    root.mkdir()
    packed = np.frombuffer(b"".join(bank.tiles), np.uint8).reshape(-1,128)
    record = load(WORK / "result.json")
    results = {}
    for mode, qs, rows in modes:
        ps, counts = [], Counter()
        for q, i in zip(qs, rows, strict=True):
            p = ds.cards[ds.card_of[i]]
            s, flag = select_visible_record(p, q, *keys)
            raw, flag2 = select_visible_record(packed[p], packed[q], *packed[keys])
            assert flag == flag2 and np.array_equal(raw, packed[s])
            ps.append(s)
            counts[flag] += 1
            counts["located_requested"] += int(flag == "located" and s[1] == ds.y[i][0])
        images, stops = generate(net, ink, mem, writing, ps, qs, bank)
        ref = [packed[ds.y[i]].tobytes() for i in rows]
        original, old_stop = archived(WORK / ("paired/"+mode+".npz"), record["archives"]["paired/"+mode+".npz"])
        old = np.array([e and x == y for x,e,y in zip(original,old_stop,ref,strict=True)])
        ok = np.array([e and x == y for x,e,y in zip(images,stops,ref,strict=True)])
        assert old.sum() == record["arms"]["paired"][mode]["exact"]
        path = root / (mode+".npz")
        image_archive(path, images, stops)
        results[mode] = dict(counts=counts, records=len(ref), exact=int(ok.sum()), original=int(old.sum()),
                     gained=int((~old & ok).sum()), lost=int((old & ~ok).sum()), stopped=sum(stops),
                     requested_form=sum(e and x[-256:-128] == y[-256:-128] for x,e,y in zip(images,stops,ref)),
                     sha256=sha(path))
    changed = {}
    ids = np.unique(dr.rows[:,1:3])
    for name, ps, qs in (("drcd", [dr.passages[dr.groups[i]] for i in ids], [dr.questions[i] for i in ids]),
                         ("dureader", du.p, du.q)):
        changed[name] = sum(not np.array_equal(select_visible_record(p,q,*keys)[0],p) for p,q in zip(ps,qs,strict=True))
    write_json(EVIDENCE/"selection.json",dict(modes=results, natural_inputs_changed=changed,
               source_sha256=sha(__file__),weights=sha(WORK/"paired/final.safetensors")))
    print(results, changed)


if __name__ == "__main__":
    main()
