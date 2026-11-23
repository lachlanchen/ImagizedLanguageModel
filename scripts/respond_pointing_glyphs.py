"""Answer a pictured passage and question with a local source-pointing image writer."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ilm.visual_lm.causal_glyph_writer import packed_page, write_strip
from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter


def render_text(text, path, index):
    from fontTools.ttLib import TTFont
    if not text:
        raise ValueError("The pictured input must contain at least one cell")
    font=ImageFont.truetype(str(path),26,index=index)
    with TTFont(path,fontNumber=index) as face:
        coverage=face.getBestCmap()
    rows=[]
    for c in text:
        im=Image.new("L",(32,32),255)
        if not c.isspace():
            if ord(c) not in coverage:
                raise ValueError(f"The selected font has no glyph for U+{ord(c):04X}")
            ink=ImageDraw.Draw(im)
            ink.text(((32-ink.textlength(c,font=font))/2,26),c,anchor="ls",font=font,fill=0)
        rows.append(np.packbits(np.asarray(im)<128))
    return torch.from_numpy(np.stack(rows))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("passage","question"):
        group=parser.add_mutually_exclusive_group(required=True)
        action="append" if key=="passage" else "store"
        group.add_argument("--"+key,action=action,type=Path,help="black-on-white horizontal strip of 32-pixel cells")
        group.add_argument("--"+key+"-text",action=action,help="render text on the host before image inference")
    parser.add_argument("--paragraph-ranker",type=Path,help="score repeated, explicitly separated passages against the pictured question")
    parser.add_argument("--model",type=Path,help="writer checkpoint directory")
    parser.add_argument("--reader",type=Path,help="optional frozen visual span reader directory")
    parser.add_argument("--selection",type=Path,help="learned choice between quoting and writing")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    parser.add_argument("--font-index",type=int,default=2)
    args=parser.parse_args()
    if args.model is None:
        candidates = (Path("writer"), Path("artifacts/bilingual_novel_v1/bilingual/page"),
                      Path("artifacts/pointing_causal_writer_v1/pointing"))
        args.model = next((p for p in candidates if (p / "pointer.safetensors").is_file()), candidates[-1])
    torch.set_num_threads(4)
    question=(packed_page(args.question) if args.question else
              render_text(args.question_text,args.font,args.font_index))
    pages=([packed_page(path) for path in args.passage] if args.passage else
           [render_text(s,args.font,args.font_index) for s in args.passage_text])
    if len(pages)>1 and args.paragraph_ranker is None:
        parser.error("Multiple passages require --paragraph-ranker")
    index=0
    if len(pages)>1:
        from ilm.visual_lm.paragraph_evidence import ParagraphRanker
        ranker=ParagraphRanker.restore(args.paragraph_ranker,args.device)
        scores=ranker.scores(question.numpy().tobytes(),[p.numpy().tobytes() for p in pages])
        index=max(range(len(scores)),key=scores.__getitem__)
        print(f"Selected paragraph {index+1} of {len(pages)}")
        del ranker
    page=pages[index]
    if bool(args.reader) != bool(args.selection):
        parser.error("--reader and --selection are used together")
    quote = None
    if args.reader:
        from ilm.visual_lm.font_invariant_reader import FontInvariantReader
        from ilm.visual_lm.pictured_answer_selection import AnswerSelection, read_options
        reader = FontInvariantReader.restore(args.reader,args.device)
        features, spans = read_options(reader, [(page.numpy().tobytes(),question.numpy().tobytes())])
        if AnswerSelection.restore(args.selection)(features).item() >= 0:
            quote = spans[0]
        del reader
    if quote is None:
        net=PointingCausalWriter.restore(args.model,args.device)
        pixels,stopped=net.answer(page,question)
    else:
        pixels=torch.from_numpy(np.frombuffer(quote,np.uint8).copy().reshape(-1,128))
        stopped=True
    write_strip(pixels.cpu().numpy(),args.output)
    print(f"{len(pixels)} returned glyphs, stopped={stopped}, action={'quote' if quote is not None else 'write'}")


if __name__=="__main__":main()
