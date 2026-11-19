"""Write a Chinese answer as a glyph strip, from pictured passages and questions."""
import argparse
from pathlib import Path
import torch

from ilm.visual_lm.causal_glyph_writer import packed_page, write_strip
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.respond_pointing_glyphs import render_text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ("passage","question"):
        g=p.add_mutually_exclusive_group(required=True)
        g.add_argument("--"+field,type=Path)
        g.add_argument("--"+field+"-text")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--model",type=Path,default=Path("artifacts/grounded_proposition_v1/planned"))
    p.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    p.add_argument("--font-index",type=int,default=2)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    torch.set_num_threads(4)
    def read(path,text):
        return packed_page(path) if path is not None else render_text(text,a.font,a.font_index)
    if (a.model/"plan.safetensors").is_file():
        from ilm.visual_lm.pictured_clause_planner import PlannedGlyphResponder
        net=PlannedGlyphResponder(a.model,a.device)
    else:
        net=PreparedGlyphResponder(a.model,a.device)
    (pixels,stopped),=net.answer_many([(read(a.passage,a.passage_text),read(a.question,a.question_text))])
    write_strip(pixels.numpy(),a.output)
    print(f"{len(pixels)} output cells, stopped={stopped}")


if __name__=="__main__":main()
