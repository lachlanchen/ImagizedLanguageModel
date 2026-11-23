"""Answer a pictured 'who' question by emitting a contiguous strip of source ink."""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.book_participant_reader import (
    BookParticipantReader, InkBank, best_strip, group_spans, padded,
)
from ilm.visual_lm.causal_glyph_writer import packed_page, write_strip
from scripts.respond_pointing_glyphs import render_text

DEFAULT = Path(__file__).resolve().parents[1]/"models/book_participant_reader/reader.safetensors"


def restore(path, device):
    net = BookParticipantReader().to(device)
    net.load_state_dict(load_file(str(path)))
    return net.eval()


@torch.inference_mode()
def answer(net, passage, question):
    device = next(net.parameters()).device
    bank = InkBank()
    p = bank.encode(passage)
    q = bank.encode(question)
    scores = net(bank.raster(device), padded([p],device), padded([q],device))
    result = best_strip(scores[0].cpu().numpy(), group_spans(p))
    return b"".join(bank.cells[t] for t in result)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ("page","question"):
        g=ap.add_mutually_exclusive_group(required=True)
        g.add_argument("--"+field, type=Path)
        g.add_argument("--"+field+"-text")
    ap.add_argument("--writer",type=Path,default=DEFAULT)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    ap.add_argument("--font-index",type=int,default=2)
    a=ap.parse_args()
    def read(field):
        path=getattr(a,field)
        x=packed_page(path) if path else render_text(getattr(a,field+"_text"),a.font,a.font_index)
        return x.numpy().tobytes()
    torch.set_num_threads(2)
    reply=answer(restore(a.writer,a.device),read("page"),read("question"))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write_strip(np.frombuffer(reply,np.uint8).reshape(-1,128),a.output)
    print(f"Wrote {len(reply)//128} source cells to {a.output}")
