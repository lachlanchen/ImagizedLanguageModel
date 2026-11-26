"""Draw the expert news comparison with original visual outputs and an editable count chart."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.util import Pt
from pptx.dml.color import RGBColor
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/'publication/ilm-nature/figures'
DATA = ROOT/'references/evidence/human_translation_calibration_v1'
ARCHIVE = ROOT/'publication/ilm-nature/evidence/wmt_news_visual_answers.zip'
W, H = 492, 216
COLORS = ['#286B62', '#98713E']


def main():
    result = json.loads((DATA/'results.json').read_text())
    writer = json.loads((DATA/'writer.json').read_text())
    assert sha256(ARCHIVE.read_bytes()).hexdigest() == writer['archive']
    index = writer['scores'][0]['index']
    with ZipFile(ARCHIVE) as z:
        r = next(r for r in json.loads(z.read('records.json')) if r['index']==index)
        pictures = [(k, z.read(f'{index}/{k}.png')) for k in ('source','answer')]
    counts = [[m[key]['joint'] for key in ('comparative','single')] for m in result['models'].values()]
    figure = FIG/'Expert_news_translation.pdf'
    pdf = Canvas(str(figure), pagesize=(W,H), invariant=1)
    deck = Presentation()
    deck.slide_width, deck.slide_height = Pt(W), Pt(H)
    s = deck.slides.add_slide(deck.slide_layouts[6])
    def label(x,y,txt,size,bold=False,*,pdf_only=False):
        pdf.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
        pdf.setFillColor(HexColor('#193731'))
        pdf.drawString(x,H-y-size,txt)
        if pdf_only:
            return
        box = s.shapes.add_textbox(Pt(x),Pt(y),Pt(min(W-x,8+pdf.stringWidth(txt,"Helvetica-Bold" if bold else "Helvetica",size))),Pt(size*1.6))
        frame = box.text_frame
        frame.auto_size=MSO_AUTO_SIZE.NONE
        frame.word_wrap=False
        frame.margin_left=frame.margin_top=frame.margin_right=frame.margin_bottom=0
        p=frame.paragraphs[0]
        p.text=txt
        p.alignment=PP_ALIGN.LEFT
        p.font.name='Arial'
        p.font.size=Pt(size)
        p.font.bold=bold
        p.font.color.rgb=RGBColor.from_string('193731')
    label(0,0,'a   Free writing from a Chinese news sentence',12,True)
    for (key,blob),top in zip(pictures,(20,44),strict=True):
        assert sha256(blob).hexdigest()==r[key]
        with Image.open(BytesIO(blob)) as im:
            width,height=(d*.4 for d in im.size)
        label(2,top,{'source':'Source','answer':'Writer'}[key],9)
        pdf.drawImage(ImageReader(BytesIO(blob)),72,H-top-height,width,height)
        s.shapes.add_picture(BytesIO(blob),Pt(72),Pt(top),Pt(width),Pt(height))
    label(0,94,'b   Agreement with professional translators on 103 news sentences',12,True)
    methods=['Candidates together, both orders','Candidates judged separately']
    labels=['Qwen3-8B','Qwen3-VL-8B']
    for i,title in enumerate(labels):
        label(4,126+i*34,title,9,True,pdf_only=True)
    for m,method in enumerate(methods):
        pdf.setFillColor(HexColor(COLORS[m]))
        pdf.rect(8+224*m,H-207,7,7,fill=1,stroke=0)
        pdf.setFont('Helvetica',8)
        pdf.drawString(19+224*m,H-207,method)
        for i,values in enumerate(counts):
            y=H-(127+i*34+m*12)
            pdf.setFillColor(HexColor(COLORS[m]))
            width=340*values[m]/103
            pdf.rect(87,y,width,7,fill=1,stroke=0)
            pdf.setFillColor(HexColor('#193731'))
            pdf.drawString(91+width,y,str(values[m]))
    # Use a PowerPoint chart with a literal Excel workbook for the same four counts.
    data=CategoryChartData()
    data.categories=labels
    for m,method in enumerate(methods):
        data.add_series(method,[values[m] for values in counts])
    chart=s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED,Pt(0),Pt(111),Pt(489),Pt(103),data).chart
    chart.has_title=False
    chart.has_legend=True
    chart.legend.position=XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout=False
    chart.legend.font.size=Pt(8)
    chart.category_axis.visible=True
    chart.category_axis.tick_label_position=XL_TICK_LABEL_POSITION.LOW
    chart.category_axis.tick_labels.font.size=Pt(9)
    chart.value_axis.minimum_scale=0
    chart.value_axis.maximum_scale=103
    chart.value_axis.has_major_gridlines=False
    chart.value_axis.visible=False
    chart.plots[0].has_data_labels=True
    chart.plots[0].data_labels.font.size=Pt(8)
    chart.category_axis.reverse_order=True
    for series,color in zip(chart.series,COLORS,strict=True):
        series.format.fill.solid()
        series.format.fill.fore_color.rgb=RGBColor.from_string(color[1:])
    pdf.showPage()
    pdf.save()
    deck.save(FIG/'Expert_news_translation.pptx')
    receipt = dict(archive=writer['archive'],index=index,counts=counts,source=r['source'],answer=r['answer'],
                   panel=result['panel'],models=list(result['models']),results=sha256((DATA/'results.json').read_bytes()).hexdigest())
    (DATA/'figure.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    main()
