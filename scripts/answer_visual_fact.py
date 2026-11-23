"""Read an image of a factual Chinese question and return an image of its answer."""
import argparse
from pathlib import Path

import numpy as np
import torch
from ilm.visual_lm.bound_fact_reader import BoundFactReader
from ilm.visual_lm.causal_glyph_writer import packed_page,write_strip
from scripts.respond_pointing_glyphs import render_text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("passage","question"):
        group=parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--"+key,type=Path)
        group.add_argument("--"+key+"-text")
    parser.add_argument("--model",type=Path,default=Path("artifacts/role_binding_v1/genuine"))
    parser.add_argument("--reader",type=Path,default=Path("artifacts/font_invariant_reader_v1"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--direct",action="store_true",help="Return only the participant or object")
    parser.add_argument("--writer",type=Path,help="Use a separately trained autoregressive image writer")
    parser.add_argument("--relation-head",type=Path,help="Optional question-side relation checkpoint")
    parser.add_argument("--participant-head",type=Path,help="Use a learned role-conditioned object span scorer")
    parser.add_argument("--word-order",choices=("subject","object","natural"),default="subject")
    parser.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    parser.add_argument("--font-index",type=int,default=2)
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    torch.set_num_threads(4)
    pair=tuple((packed_page(file) if file else render_text(text,args.font,args.font_index)).numpy().tobytes()
               for file,text in ((args.passage,args.passage_text),(args.question,args.question_text)))
    model=BoundFactReader(args.reader,args.model,args.device)
    if args.relation_head is None and (args.model/"pictured-relation.npz").is_file():
        args.relation_head=args.model/"pictured-relation.npz"
    if args.participant_head and not args.relation_head:
        parser.error("--participant-head requires pictured relation weights in the model or --relation-head")
    if args.relation_head:
        from ilm.visual_lm.pictured_relation import PicturedRelation
        channel=PicturedRelation(args.relation_head)
        if args.participant_head:
            from ilm.visual_lm.role_conditioned_span import ParticipantReader
            fields=ParticipantReader(model,channel,args.participant_head).fields_many([pair])
        else:
            fields=[(s,channel.choose(pair[1],s,r)[0],o) for s,r,o in model.fields_many([pair])]
    else:
        fields=model.fields_many([pair])
    if args.writer:
        if args.direct:parser.error("--direct and --writer select different output modes")
        from ilm.visual_lm.fact_realizer import FactRealizer
        writer=FactRealizer(args.writer,args.device)
        (output,stopped),=writer.answer_fields(fields,args.word_order)
        print(f"Writer stopped: {stopped}")
    else:
        if args.word_order!="subject":parser.error("Changing word order requires --writer")
        from ilm.visual_lm.bound_fact_reader import render_fact
        (s,r,o),=fields
        output=o if args.direct else render_fact(model.frames[r],s,o)
    write_strip(np.frombuffer(output,np.uint8).reshape(-1,128),args.output)


if __name__=="__main__":
    main()
