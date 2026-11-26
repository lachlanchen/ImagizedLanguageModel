"""Make editable book and translation plates from measured scores and archived PNGs."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import fitz
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.util import Pt
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'publication/ilm-nature'
FIG=PAPER/'figures'
PUB=ROOT/'references/evidence/book_order_prior_v1'


class Plate:
    def __init__(self,name,height):
        self.name=name
        self.height=height
        self.pdf=Canvas(str(FIG/(name+'.pdf')),pagesize=(492,height),invariant=1)
        self.deck=Presentation()
        self.deck.slide_width,self.deck.slide_height=Pt(492),Pt(height)
        self.slide=self.deck.slides.add_slide(self.deck.slide_layouts[6])
        self.sources=[]

    def text(self,x,y,s,size=9,bold=False,color='183631'):
        p=self.pdf
        font='Helvetica-Bold' if bold else 'Helvetica'
        p.setFillColor(HexColor('#'+color));p.setFont(font,size);p.drawString(x,self.height-y-size,s)
        box=self.slide.shapes.add_textbox(Pt(x),Pt(y),Pt(492-x),Pt(size*1.5))
        frame=box.text_frame;frame.auto_size=MSO_AUTO_SIZE.NONE;frame.word_wrap=False
        frame.margin_top=frame.margin_bottom=frame.margin_left=frame.margin_right=0
        par=frame.paragraphs[0];par.alignment=PP_ALIGN.LEFT;par.text=s
        par.font.name='Arial';par.font.size=Pt(size);par.font.bold=bold;par.font.color.rgb=RGBColor.from_string(color)

    def strip(self,z,member,expected,label,y):
        blob=z.read(member)
        assert sha256(blob).hexdigest()==expected
        with Image.open(BytesIO(blob)) as im:width,height=im.size
        scale=min(.37,374/width)
        h=height*scale;w=width*scale
        self.text(7,y+1,label)
        self.pdf.drawImage(ImageReader(BytesIO(blob)),113,self.height-y-h,w,h)
        self.slide.shapes.add_picture(BytesIO(blob),Pt(113),Pt(y),Pt(w),Pt(h))
        self.sources.append(dict(archive=Path(z.filename).name,member=member,sha256=expected))
        return y+max(13,h)+3

    def finish(self):
        self.pdf.showPage();self.pdf.save();self.deck.save(FIG/(self.name+'.pptx'))
        with fitz.open(FIG/(self.name+'.pdf')) as doc:
            doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(FIG/(self.name+'.png'))


def books():
    result=json.loads((PUB/'prose.json').read_text())
    keys=['baseline','scrambled','intact']
    categories=['No book pretraining','Scrambled words','Intact word order']
    series={label:[result['scores'][arm+'_'+stage]['bits'] for arm in keys]
            for label,stage in [('Book prediction','prior'),('After bilingual training','adapted')]}
    plate=Plate('Book_order_learning',227)
    plate.text(0,0,'a   Predicting the next glyph on 16 unseen books',11,True)
    plate.text(7,18,'Bits per glyph',8)
    # The PDF and its editable PowerPoint chart use the same unrounded measurements.
    colors=['24766D','BD8452']
    p=plate.pdf
    for i,(name,values) in enumerate(series.items()):
        p.setFont('Helvetica',8);p.setFillColor(HexColor('#'+colors[i]));p.drawString(231+103*i,plate.height-24,name)
        for j,(label,val) in enumerate(zip(categories,values,strict=True)):
            y=37+j*21+i*7
            p.setFillColor(HexColor('#'+colors[i]));p.rect(109,plate.height-y-5,67*val,5,fill=1,stroke=0)
            p.setFont('Helvetica',7);p.drawString(113+67*val,plate.height-y-5,f'{val:.2f}')
            if i==0:
                p.setFillColor(HexColor('#183631'));p.setFont('Helvetica',9);p.drawString(7,plate.height-y-9,label)
    data=CategoryChartData();data.categories=categories
    for label,values in series.items():data.add_series(label,values)
    chart=plate.slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED,Pt(10),Pt(27),Pt(466),Pt(82),data).chart
    chart.has_legend=True;chart.legend.position=XL_LEGEND_POSITION.RIGHT;chart.legend.font.size=Pt(8)
    chart.value_axis.minimum_scale=0;chart.value_axis.maximum_scale=4.5
    chart.value_axis.tick_labels.font.size=Pt(7);chart.category_axis.tick_labels.font.size=Pt(8)
    chart.category_axis.reverse_order=True
    for s,c in zip(chart.series,colors,strict=True):
        s.format.fill.solid();s.format.fill.fore_color.rgb=RGBColor.from_string(c)
    plate.text(0,114,'b   A free continuation from the first held prefix',11,True)
    with ZipFile(PAPER/'evidence/book_order_continuations.zip') as z:
        row=json.loads(z.read('records.json'))[0]
        y=133
        for k,name in [('prefix','Given prefix'),('reference','Published continuation'),('intact_prior','Intact-order fit'),('scrambled_prior','Scrambled-order fit')]:
            item=row[k] if k in row else row['answers'][k]
            y=plate.strip(z,f"{row['window']}/{k}.png",item['sha256'],name,y)
    plate.finish()
    return dict(file=plate.name,items=plate.sources,series=series,categories=categories)


def bilingual():
    result=json.loads((PUB/'meaning.json').read_text())
    with ZipFile(PAPER/'evidence/book_order_translations.zip') as z:
        rows=json.loads(z.read('records.json'))
        news=next(i for i,r in enumerate(rows[:128]) if r['open'] and not result['news'][i]['intact'])
        winner=next(i for i,f in enumerate(result['edit_families']) if f['intact'] and
                    all(rows[128+2*i+s]['answers']['intact']['exact'] for s in (0,1)))
        a=128+winner*2
        parts=[('a   First edit family with two exact answers',
               [(a,'source','Chinese source A'),(a,'intact','Free answer A'),(a+1,'source','Chinese source B'),(a+1,'intact','Free answer B')]),
               ('b   An unfamiliar source from held news',
               [(news,'source','Chinese source'),(news,'reference','Supplied reference'),(news,'baseline','No book pretraining'),(news,'intact','Intact-order fit')])]
        height=0
        for _,items in parts:
            height+=26
            for i,k,_ in items:
                with Image.open(BytesIO(z.read(f'{i}/{k}.png'))) as im:w,h=im.size
                height+=max(13,h*min(.37,374/w))+3
        plate=Plate('Book_order_translation',height)
        y=0
        for title,items in parts:
            plate.text(0,y,title,11,True);y+=19
            for i,k,name in items:
                r=rows[i]
                info=r['input' if k=='source' else k] if k in ('source','reference') else r['answers'][k]
                y=plate.strip(z,f'{i}/{k}.png',info['sha256'],name,y)
            y+=7
        plate.finish()
    return dict(file=plate.name,items=plate.sources,news=news,family=rows[a]['family'],credited=result['edit_families'][winner]['intact'])


if __name__=='__main__':
    results=[books(),bilingual()]
    (PUB/'figures.json').write_text(json.dumps(results,indent=2)+'\n')
