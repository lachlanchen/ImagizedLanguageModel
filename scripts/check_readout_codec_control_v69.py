#!/usr/bin/env python3
"""Post-hoc supplied-code control on all fixed V69 first-block images."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from scripts.probe_conditional_readout_v69 import ROOT, atomic_json, cpu_only, identity, sha, source_bound_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();cpu_only();begun=time.perf_counter()
    revision,sources,_,rows=source_bound_rows(args.run)
    verification=json.loads((args.run/"verification.json").read_text())
    if (not verification["offline_checks_pass"]
            or sha(args.run/"fit_report.json")!=verification["fit_report_sha256"]):
        raise ValueError("completed verified readout measurements required")
    args.out.mkdir(parents=True,exist_ok=False)
    writer=CodecBlockWriter(load_qualified_writing_codec(
        ROOT/"artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt",torch.device("cpu")))
    measured,raw=[],[]
    with torch.no_grad():
        for start in range(0,len(rows),32):
            reference=torch.cat([r["reference"] for r in rows[start:start+32]])
            pixels=writer.predict(writer.encode_targets(reference));raw.append(pixels)
            for j,value in enumerate(pixels):
                a=value.numpy()<0;b=reference[j].numpy()<0;denominator=int(a.sum()+b.sum())
                measured.append(dict(index=start+j,foreground_f1=2*int((a&b).sum())/denominator if denominator else 1.,
                    exact=bool(np.array_equal(a,b))))
    groups={}
    for split in ("train","development"):
        for source in ("cmrc","dolly"):
            selected=[m for m in measured if rows[m["index"]]["split"]==split and rows[m["index"]]["source"]==source]
            groups[split+"/"+source]=dict(count=len(selected),
                foreground_f1=float(np.mean([m["foreground_f1"] for m in selected])),
                exact_count=sum(m["exact"] for m in selected))
    torch.save(torch.cat(raw),args.out/"pixels.pt")
    atomic_json(args.out/"report.json",dict(source_commit=revision,source_code_sha256=sha(Path(__file__)),
        source_verification_sha256=sha(args.run/"verification.json"),raw_pixels_sha256=sha(args.out/"pixels.pt"),
        post_hoc=True,used_for_head_selection=False,cpu_codec_blocks=704,full_actor_calls=0,
        optimizer_updates=0,gpu_allocations=0,groups=groups,measurements=measured,
        elapsed_seconds=time.perf_counter()-begun,language_quality_claim=False))
    print(json.dumps(groups,indent=2))


if __name__=="__main__":
    main()
