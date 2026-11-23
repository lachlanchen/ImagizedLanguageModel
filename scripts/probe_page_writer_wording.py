"""Replay the frozen date boards with the date phrasing used during adaptation."""
import argparse
import json
from pathlib import Path
import torch

from scripts.evaluate_page_evidence_writer import generate, score
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.grounded_proposition_corpus import noun
from scripts.page_evidence_corpus import DEFAULT
from scripts.prepare_visual_facts import Face, FONTS
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner


def familiar(x):
    return f"先说明{noun(x['s'],x['r'])}，再写同段提到的年份。"


def probe(folder):
    torch.set_num_threads(2)
    dest=folder/"familiar"
    dest.mkdir(exist_ok=False)
    data_path=folder/"held.jsonl"
    if sha(data_path)!=json.loads((folder/"corpus.json").read_text())["held"]["sha256"]:
        raise ValueError("Held inputs changed")
    face=Face((FONTS[0],2))
    data=[dict(x,q=familiar(x)) for x in map(json.loads,data_path.read_text().splitlines())
          if x["task"] in ("single","dual")]
    if not all(sha(folder/"page"/n)==h for n,h in json.loads((folder/"page_fit.json").read_text())["weights"].items()):
        raise ValueError("Writer changed")
    outputs=generate(folder/"page",data,"page",face)
    stats=score(data,outputs,face)
    store_streams(dest/"streams.npz",[y for y,_ in outputs],[s for _,s in outputs])
    record(dest/"results.json",dict(groups=stats,streams=sha(dest/"streams.npz")))
    print(stats,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",type=Path,default=DEFAULT)
    with gpu_owner():
        probe(ap.parse_args().run_dir)
