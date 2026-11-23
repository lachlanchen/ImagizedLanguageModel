"""Answer two pictured questions in order, writing a new stopped glyph image."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.grounded_ink_editor import GroundedInkEditor
from ilm.visual_lm.optical_role_reader import answer,restore
from ilm.visual_lm.portable_writer import read_image
from scripts.read_optical_role import digest
from scripts.respond_pointing_glyphs import render_text

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/"models/grounded_ink_editor"


def load(base,device):
    optical=ROOT/"models/optical_role_reader"
    spec=json.loads((optical/"weights.json").read_text())
    adapter=optical/"adapter.safetensors"
    editor=RELEASE/"editor.safetensors"
    for p,h in [(base/n,h) for n,h in spec["base"].items()]+[(adapter,spec["weights"]),(editor,json.loads((RELEASE/"weights.json").read_text())["editor"])]:
        if not p.is_file() or digest(p)!=h:raise ValueError(f"Missing or different model file: {p}")
    writer=GroundedInkEditor().to(device).eval();writer.load_state_dict(load_file(str(editor)))
    return restore(base,device,adapter),writer


def respond(reader,writer,page,questions,style):
    regions=[]
    for q in questions:
        raw=answer(reader,page,q)
        a=next(i for i in range(0,len(page),128) if page.startswith(raw,i))//128
        regions.append((a,a+len(raw)//128-1))
    return writer.emit(page,regions,style)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ("page","first","second","instruction"):
        g=ap.add_mutually_exclusive_group(required=True)
        g.add_argument("--"+field,type=Path)
        g.add_argument("--"+field+"-text")
    ap.add_argument("--page-cells",type=int,help="Valid row-major cell count in a padded page grid")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--base",type=Path,default=ROOT/"artifacts/font_invariant_reader_v1")
    ap.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    ap.add_argument("--font-index",type=int,default=2)
    ap.add_argument("--device",default="cpu")
    a=ap.parse_args()
    if a.page_cells and not a.page:ap.error("--page-cells requires an image")
    def read(field):
        p=getattr(a,field)
        if p:return read_image(p,a.page_cells if field=="page" else None).tobytes()
        return render_text(getattr(a,field+"_text"),a.font,a.font_index).numpy().tobytes()
    torch.set_num_threads(2);reader,editor=load(a.base,a.device)
    output,end=respond(reader,editor,read("page"),[read("first"),read("second")],read("instruction"))
    a.output.parent.mkdir(exist_ok=True,parents=True)
    write_strip(np.frombuffer(output,np.uint8).reshape(-1,128),a.output)
    print(json.dumps(dict(cells=len(output)//128,stopped=end)))


if __name__=="__main__":
    main()
