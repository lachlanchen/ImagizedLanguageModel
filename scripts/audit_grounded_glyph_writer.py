"""Recount held-out scores from physical images and evaluate fixed, publishable questions."""
from collections import Counter
import numpy as np
import torch

from scripts.audit_composed_writer_outputs import distance
from scripts.illustrate_composed_writer import EXAMPLES, packed
from scripts.infer_pictured_entry import page
from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.read_grounded_glyph_answer import respond
from scripts.train_grounded_glyph_writer import DEST, ROOT, SEED, WORK, Split, load_writer
from scripts.train_paired_passage import unpack_sequences


def paired_interval(a, b):
    delta = np.array(a, dtype=float) - b
    draws = np.random.default_rng(SEED).integers(len(a), size=(10000, len(a)))
    return dict(gains=int((delta>0).sum()), losses=int((delta<0).sum()), mean=float(delta.mean()),
                ci95=np.quantile(delta[draws].mean(1), [.025,.975]).tolist())


def recount(ds, tiles, arm, mode):
    cases = load(WORK/arm/(mode+".json"))
    streams = unpack_sequences(WORK/arm/(mode+".npz"))
    totals = Counter()
    checks = []
    blank = bytes(128)
    for i, (ids, case) in enumerate(zip(streams, cases, strict=True)):
        ref, p, q = [list(map(bytes, ds.tiles[s[i]])) for s in (ds.y, ds.p, ds.q)]
        out = list(map(bytes, tiles[ids]))
        visible = set(p+q)
        prefix = min(len(out), len(ref))
        for j, (a, b) in enumerate(zip(out, ref)):
            if a != b:
                prefix = j
                break
        # Partition errors into wrong content, premature ending and surplus ink.
        failure = "exact" if out == ref and case["stopped"] else (
            "extra_after_reference" if prefix == len(ref) and len(out)>len(ref) else
            "ends_before_reference" if prefix == len(out) and len(out)<len(ref) else
            "matches_without_stop" if out == ref else "different_glyph")
        truth = dict(exact=int(bool(case["stopped"]) and ref==out), first=int(ref[:1]==out[:1]),
                     first4=int(ref[:4]==out[:4]), output_cells=len(out), all_cells=len(ref),
                     absent_match=sum((Counter(s for s in out if s not in visible) &
                                       Counter(s for s in ref if s not in visible)).values()),
                     edit=distance(ref,out))
        assert all(case[k]==v for k,v in truth.items())
        totals[failure] += 1
        totals["entire_passage"] += int(out == p)
        totals["diverges_before_new_content"] += int(case["group"] == "absent_content_ink" and
                prefix < next(j for j, t in enumerate(ref) if t not in visible))
        totals.update(dict(empty=int(not out), all_white=int(bool(out) and set(out)=={blank}),
            reaches_cap=int(len(out)==64), repeats_last_glyph_eight=int(len(out)>=8 and len(set(out[-8:]))==1),
            copies_passage_opening4=int(len(out)>=4 and out[:4]==p[:4]),
            correct_opening_then_wrong=int(truth["first4"] and not truth["exact"])))
        checks.append(truth["first4"])
    return totals, checks


def main():
    torch.set_num_threads(2)
    report = load(WORK/"results.json")
    source = ROOT/"references/evidence/grounded_glyph_writer_20261119"
    source.mkdir(exist_ok=True)
    ds = Split("development")
    tiles = np.load(WORK/"evaluation_tiles.npy")
    diagnostics, openings, fits = {}, {}, {}
    memory = np.load(WORK/"memory.npy")
    for arm in report:
        fits[arm] = load(WORK/arm/"fit.json")
        for mode in report[arm]:
            diag, first = recount(ds, tiles, arm, mode)
            diagnostics[arm+"_"+mode] = diag
            openings[arm+"_"+mode] = first
    tests = {c:paired_interval(openings["paired_original"], openings[c]) for c in
             ("deranged_original","paired_rotated")}
    write_json(source/"audit.json", dict(counts=report,diagnostics=diagnostics,first_four_contrasts=tests,
               fits=fits, memory_tiles=len(memory), memory_sha256=sha(WORK/"memory.npy"),
               preparation=load(DEST/"receipt.json")))
    net = load_writer(WORK/"paired","cpu")
    examples, dims = {}, {}
    for name, passage, question, answer in EXAMPLES:
        raw, stopped = respond(net,memory,packed(passage),packed(question),"cpu")
        reference = packed(answer).tobytes()
        is_passage = raw == packed(passage).tobytes()
        examples[name] = dict(copies_entire_passage=is_passage,cells=len(raw)//128, stopped=stopped, exact=stopped and raw==reference,
                              distinct_shapes=len(set(raw[i:i+128] for i in range(0,len(raw),128))))
        for suffix, data in (("output",raw),("reference",reference)):
            img = page(data)
            key = name+"_"+suffix
            img.save(source/(key+".png"))
            dims[key] = list(img.size)
    trials = [("Matched","paired","original"),("Question rotated","paired","rotated"),
              ("Pairs broken","deranged","original"),("Both changed","deranged","rotated")]
    table = [["Training / test","First glyph","First four","Complete","Input-absent"]]
    for name, train, test in trials:
        row = report[train][test]
        table.append([name, *map(str,[row["all"]["first"],row["all"]["first4"],row["all"]["exact"],
                                    row["absent_content_ink"]["exact"]])])
    write_json(source/"figure.json",dict(table=table, series=[[report[a][b]["all"]["first4"]
               for _,a,b in trials]], dimensions=dims,examples=examples))
    print(table, tests, diagnostics, examples)


if __name__ == "__main__":
    main()
