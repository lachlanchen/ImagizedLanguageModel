"""Use an installed visual span reader to condition a local sentence writer."""
import argparse
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.causal_glyph_writer import packed_page,write_strip
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.pictured_answer_selection import read_options
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.respond_pointing_glyphs import render_text


class EvidenceResponder:
    def __init__(self,writer,reader,device):
        self.writer=PreparedGlyphResponder(writer,device)
        self.reader=FontInvariantReader.restore(reader,device)

    @torch.inference_mode()
    def answer_many(self,pairs):
        if not pairs:return []
        raw=[(p.cpu().numpy().tobytes(),q.cpu().numpy().tobytes()) for p,q in pairs]
        _,spans=read_options(self.reader,raw)
        marked=[torch.from_numpy(np.frombuffer(e+bytes(128)+p,np.uint8).copy().reshape(-1,128)) for (p,_),e in zip(raw,spans)]
        return self.writer.answer_many([(p,q) for p,(_,q) in zip(marked,pairs)])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for field in ("passage","question"):
        g=p.add_mutually_exclusive_group(required=True)
        g.add_argument("--"+field,type=Path)
        g.add_argument("--"+field+"-text")
    p.add_argument("--model",type=Path,default=Path("artifacts/grounded_proposition_v1/evidence"))
    p.add_argument("--reader",type=Path,default=Path("artifacts/font_invariant_reader_v1"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    p.add_argument("--font-index",type=int,default=2)
    a=p.parse_args()
    torch.set_num_threads(4)
    pair=tuple(packed_page(file) if file else render_text(text,a.font,a.font_index)
               for file,text in ((a.passage,a.passage_text),(a.question,a.question_text)))
    (pixels,stopped),=EvidenceResponder(a.model,a.reader,a.device).answer_many([pair])
    write_strip(pixels.numpy(),a.output)
    print(f"{len(pixels)} output cells, stopped={stopped}")
