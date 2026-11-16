#!/usr/bin/env python3
"""Render affected pages for actual review; this receipt does not claim inspection."""
import json
import argparse
from pathlib import Path
import resource
import time
import fitz
from PIL import Image,ImageDraw,ImageFont
from build_ranking_completed_article import host,sha
from prepare_glyph_score_gap_figure import owners
from prepare_visual_context_evidence import ROOT,protected_sources

STAGE=ROOT/'artifacts/visual_context_evidence_publication_v1/article_attempt02'
TARGETS={'manuscript':('reading',[1,32,33,34,40,42,43,44,54,55,56]),
 'manuscript_review':('review',[1,34,35,36,37,38,39,40,63,64,65,66,67,98,100,101]),
 'redlined_manuscript':('redlines/redlined_manuscript',[1,2,35,36,37,38,39,40,41,65,66,67,68,99,101,102])}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=2);args=parser.parse_args()
    STAGE=STAGE.parent/f'article_attempt{args.attempt:02}'
    began=time.perf_counter();protected=owners();protected_sources();assert host()['available_gib']>=48
    output=STAGE/'detailed_review';output.mkdir(exist_ok=False);results={}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for name,(folder,pages) in TARGETS.items():
        file=STAGE/folder/(name+'.pdf');pairs=[]
        with fitz.open(file) as pdf:
            for index in range(0,len(pages),2):
                assert host()['available_gib']>=48 and time.perf_counter()-began<180
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2<2
                numbers=pages[index:index+2];images=[]
                for p in numbers:
                    pix=pdf[p-1].get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
                    images.append(Image.frombytes('RGB',(pix.width,pix.height),pix.samples))
                board=Image.new('RGB',(sum(i.width for i in images),max(i.height for i in images)+35),'white')
                draw=ImageDraw.Draw(board);x=0
                for p,image in zip(numbers,images):
                    board.paste(image,(x,35));draw.text((x+8,6),f'{name} / page {p}',font=font,fill='black');x+=image.width
                target=output/f'{name}-{index//2+1:02}.png';board.save(target)
                pairs.append(dict(file=str(target.relative_to(ROOT)),pages=numbers,sha256=sha(target)))
        results[name]=dict(pdf_sha256=sha(file),pages=pages,pairs=pairs)
    protected_sources();assert owners()==protected
    (output/'manifest.json').write_text(json.dumps(dict(documents=results,seconds=time.perf_counter()-began,
        all_pages_actually_viewed=False,source_sha256=sha(__file__)),indent=2)+'\n')
    print(json.dumps(dict(pages=sum(len(v[1]) for v in TARGETS.values()),seconds=time.perf_counter()-began)))
