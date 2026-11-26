"""Publish authentic input and output bitmaps from the matched reviewed-news trial."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import fitz
from PIL import Image
from pptx import Presentation
from pptx.util import Pt
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.dml.color import RGBColor
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

ROOT=Path(__file__).resolve().parents[1]
PUB=ROOT/'references/evidence/open_news_external_v1'
FIG=ROOT/'publication/ilm-nature/figures'
ARCHIVE=ROOT/'publication/ilm-nature/evidence/reviewed_news_writers.zip'


def main():
    result=json.loads((PUB/'meaning.json').read_text())
    assert sha256(ARCHIVE.read_bytes()).hexdigest()==result['archive']
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))
        scope=[i for i,r in enumerate(rows) if r['open']]
        score=result['grades']
        first=next(i for i,r in enumerate(rows) if not r['open'] and score[i]['ink'])
        second=next(i for i in scope if not score[i]['ink'])
        parts=[];items=[]
        for i,title in [(first,'a   A credited translation with familiar shapes'),
                        (second,'b   An uncredited translation with an unfamiliar shape')]:
            parts.append((title,[]))
            for key,label in [('source','Source'),('reference','Reference'),('parent','Before training'),('ink','After training')]:
                blob=z.read(f'{i}/{key}.png')
                hsh=sha256(blob).hexdigest()
                assert hsh==(rows[i][key] if key in ('source','reference') else rows[i]['answers'][key]['sha256'])
                with Image.open(BytesIO(blob)) as im:w,h=im.size
                parts[-1][1].append((blob,label,w*.35,h*.35))
                items.append(dict(index=i,key=key,sha256=hsh))
    height=sum(22+sum(max(11,h)+3 for _,_,_,h in material)+4 for _,material in parts)
    w=492
    pdf=Canvas(str(FIG/'Reviewed_news_writer.pdf'),pagesize=(w,height),invariant=1)
    deck=Presentation();deck.slide_width=Pt(w);deck.slide_height=Pt(height)
    s=deck.slides.add_slide(deck.slide_layouts[6])
    def mark(x,y,text,size,bold=False):
        pdf.setFillColor(HexColor('#1b3531'));pdf.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
        pdf.drawString(x,height-y-size,text)
        box=s.shapes.add_textbox(Pt(x),Pt(y),Pt(min(w-x,8+pdf.stringWidth(text,"Helvetica-Bold" if bold else "Helvetica",size))),Pt(1.5*size))
        f=box.text_frame;f.auto_size=MSO_AUTO_SIZE.NONE;f.word_wrap=False;f.margin_left=f.margin_right=f.margin_top=f.margin_bottom=0
        p=f.paragraphs[0];p.text=text;p.alignment=PP_ALIGN.LEFT;p.font.name='Arial';p.font.size=Pt(size);p.font.bold=bold
        p.font.color.rgb=RGBColor.from_string('1B3531')
    y=0
    for title,material in parts:
        mark(0,y,title,11,True);y+=19
        for blob,label,width,h in material:
            mark(0,y+1,label,9)
            pdf.drawImage(ImageReader(BytesIO(blob)),89,height-y-h,width,h)
            s.shapes.add_picture(BytesIO(blob),Pt(89),Pt(y),Pt(width),Pt(h))
            y+=max(11,h)+3
        y+=7
    pdf.showPage();pdf.save();deck.save(FIG/'Reviewed_news_writer.pptx')
    with fitz.open(FIG/'Reviewed_news_writer.pdf') as document:
        document[0].get_pixmap(matrix=fitz.Matrix(2.5,2.5)).save(FIG/'Reviewed_news_writer.png')
    (PUB/'figure.json').write_text(json.dumps(dict(archive=result['archive'],items=items,
        selection="first credited familiar and first uncredited unfamiliar",meaning=sha256((PUB/'meaning.json').read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':
    main()
