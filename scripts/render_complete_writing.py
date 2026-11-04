#!/usr/bin/env python3
"""Render complete typed writing as a native image-language prompt strip."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from ilm.visual_lm.full_text_renderer import CompleteLineRenderer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text",required=True);parser.add_argument("--out",required=True)
    parser.add_argument("--maximum-blocks",type=int,default=240)
    args=parser.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError("choose a new output image")
    renderer=CompleteLineRenderer();image,metadata=renderer.render(args.text,args.maximum_blocks)
    out.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(image,mode="L").save(out)
    receipt=dict(layout=metadata,renderer=renderer.provenance,
        source_text_sha256=hashlib.sha256(args.text.encode()).hexdigest(),
        model_calls=0,role="host input adapter; language quality requires model evaluation")
    out.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(metadata))


if __name__=="__main__":main()
