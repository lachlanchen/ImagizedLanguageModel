"""Generate and score complete, stopped replies to pictured requests and source edits."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.page_evidence_corpus import DEFAULT
from scripts.prepare_visual_facts import Face, FONTS
from scripts.train_causal_glyph_writer import in_cells, record, sha
from scripts.train_fact_realizer import COMMANDS
from scripts.train_page_evidence_writer import BASE, source
from scripts.train_span_choice import gpu_owner


def tensor(raw):
    return torch.from_numpy(np.frombuffer(raw, np.uint8).copy().reshape(-1, 128))


def contains(raw, part):
    return any(raw[i:i+len(part)] == part for i in range(0, len(raw)-len(part)+1, 128))


def score(data, outputs, face):
    groups, boards = defaultdict(Counter), defaultdict(list)
    for v, (raw, stopped) in zip(data, outputs, strict=True):
        g = groups[v["task"]]
        exact = stopped and raw == face(v["y"])
        g["rows"] += 1
        g["stopped"] += int(stopped)
        g["exact"] += int(exact)
        g["both_names_and_stop"] += int(stopped and contains(raw,face(v["s"])) and contains(raw,face(v["o"])))
        date_ok = False
        if v["year"]:
            date_ok = stopped and contains(raw,face(v["year"])) and not contains(raw,face(v["rival"]))
            g["correct_year_and_stop"] += int(date_ok)
        boards[v["task"], v["board"]].append((exact,date_ok))
    for (task, _), rows in boards.items():
        expected = {"duie":2, "cmrc":2, "dual":4, "single":2, "clause":1}[task]
        if len(rows) != expected:
            raise ValueError(f"Incomplete {task} test group")
        groups[task]["boards"] += 1
        groups[task]["complete"] += int(all(a for a, _ in rows))
        if task in ("dual", "single"):
            groups[task]["all_years"] += int(all(b for _, b in rows))
    return dict(groups)


@torch.inference_mode()
def generate(folder, data, arm, face):
    net = PreparedGlyphResponder(folder, "cuda")
    output = [None]*len(data)
    def request(v):
        return COMMANDS[v["request"]] if arm == "initial" else v["q"]
    def passage(v):
        return source(v, "page" if arm == "page" else "fields")
    ordered = sorted(range(len(data)), key=lambda i:len(passage(data[i]))+2*len(request(data[i])))
    for start in range(0,len(ordered),16):
        take = ordered[start:start+16]
        args = [(tensor(face(passage(data[i]))),tensor(face(request(data[i])))) for i in take]
        for i,(bitmap,stop) in zip(take,net.answer_many(args),strict=True):
            output[i] = (bitmap.numpy().tobytes(),stop)
    return output


def evaluate(folder):
    torch.set_num_threads(2)
    root = folder / "evaluation"
    root.mkdir(exist_ok=False)
    corpus = json.loads((folder/"corpus.json").read_text())
    if sha(folder/"held.jsonl") != corpus["held"]["sha256"]:
        raise ValueError("Evaluation passages changed")
    data = [json.loads(s) for s in (folder/"held.jsonl").read_text().splitlines()]
    face = Face((FONTS[0],2))
    results, samples, detail = {}, {}, {}
    t = monotonic()
    for arm in ("page","fields","initial"):
        cases = [v for v in data if arm != "initial" or v["task"] in ("duie","cmrc")]
        weights = BASE if arm == "initial" else folder/arm
        hashes = corpus["base"] if arm == "initial" else json.loads((folder/(arm+"_fit.json")).read_text())["weights"]
        if any(sha(weights/n)!=h for n,h in hashes.items()):
            raise ValueError(f"{arm} weights differ from fit")
        output = generate(weights,cases,arm,face)
        stats = score(cases,output,face)
        raw_path = root/(arm+".npz")
        store_streams(raw_path,[r for r,_ in output],[s for _,s in output])
        results[arm] = dict(groups=stats,streams=sha(raw_path))
        if arm == "page":
            single = next(i for i,v in enumerate(cases) if v["task"]=="single")
            dual = next(i for i,v in enumerate(cases) if v["task"]=="dual")
            failed = next((i for i,v in enumerate(cases) if v["task"]=="dual" and output[i]!=(face(v["y"]),True)),None)
            samples = dict(single_original=single,single_edit=single+1,dual_first=dual)
            if failed is not None:
                samples["dual_first_error"] = failed
            legend = {face.cell(c):c for v in cases for text in (v["p"],v["q"],v["y"]) for c in text}
            for name,i in samples.items():
                v = cases[i]
                raw,stop = output[i]
                detail[name] = dict(index=i, board=v["board"], q=v["q"], reference=v["y"],
                                    transcription="".join(legend.get(c,"□") for c in in_cells(raw)),stopped=stop)
                for key, ink in (("page",face(v["p"])),("question",face(v["q"])),("output",raw),
                                 ("reference",face(v["y"]))):
                    write_strip(tensor(ink).numpy(), root/(name+"_"+key+".png"))
        print(arm, stats, flush=True)
    checks = {}
    for group in ("single","dual","duie","cmrc"):
        page,control = [results[a]["groups"][group] for a in ("page","fields")]
        if group in ("single","dual"):
            checks[group] = 2*page["complete"]>=page["boards"] and page["complete"]>control["complete"]
        else:
            checks[group] = 10*page["complete"]>=9*control["complete"]
    record(root/"examples.json",detail)
    record(root/"results.json",dict(arms=results,requirements=checks,seconds=monotonic()-t))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=DEFAULT)
    with gpu_owner():
        evaluate(parser.parse_args().run_dir)
