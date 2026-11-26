"""Draw original held answer images and longitudinal scores from both completed fits."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile
import fitz
from PIL import Image
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
from scripts.evaluate_visual_training_breadth import PUB,MODEL,ROOT,ARCHIVE,base
from scripts.figure_sentence_reconstruction import draw_plate

FIG=ROOT/'publication/ilm-nature/figures'


def learning_curve():
    start=base.read(PUB/'starting-development.json')
    series={}
    for t,lang in enumerate(('English','Chinese')):
        for arm in ('broad','narrow'):
            fit=base.read(MODEL/arm/'fit.json')
            values=[start[t]['mean_glyph_f2']]+[ep['development'][t]['mean_glyph_f2'] for ep in fit['epochs']]
            series[f'{lang} / {arm}']=values
    path=FIG/'Training_breadth_curves.pdf'
    canvas=Canvas(str(path),pagesize=(490,174),invariant=1)
    for t,lang in enumerate(('English','Chinese')):
        left=34+245*t
        width=192
        low,high=(.34,.46) if t==0 else (.11,.23)
        levels=[v for k,row in series.items() if lang in k for v in row]
        low=min(low,int(min(levels)*100//2)*.02)
        high=max(high,(int(max(levels)*100//2)+1)*.02)
        y=lambda value: 37+96*(value-low)/(high-low)
        canvas.setFont('Helvetica-Bold',10)
        canvas.setFillColor(HexColor('#19332f'))
        canvas.drawString(left,150,'Into '+lang)
        canvas.setFont('Helvetica',7.5)
        for step in range(4):
            v=low+(high-low)*step/3
            canvas.setStrokeColor(HexColor('#d8dfde'))
            canvas.setLineWidth(.45)
            canvas.line(left,y(v),left+width,y(v))
            canvas.drawRightString(left-5,y(v)-2,f'{v:.2f}')
        for ep in range(7):
            canvas.drawCentredString(left+width*ep/6,26,str(ep))
        canvas.drawCentredString(left+width/2,13,'Training pass')
        for arm,color in (('broad','#126b62'),('narrow','#975733')):
            canvas.setStrokeColor(HexColor(color));canvas.setFillColor(HexColor(color))
            canvas.setDash([] if arm=='broad' else [3,2]); canvas.setLineWidth(1.45)
            pts=[(left+width*i/6,y(v)) for i,v in enumerate(series[lang+' / '+arm])]
            for a,b in zip(pts,pts[1:]):
                canvas.line(*a,*b)
            canvas.setDash([])
            for x,yy in pts:
                canvas.circle(x,yy,1.75,fill=int(arm=='broad'))
    canvas.setFont('Helvetica',7.8)
    for x,color,label in ((133,'#126b62','Broad corpus'),(260,'#975733','Repeated corpus')):
        canvas.setFillColor(HexColor(color));canvas.drawString(x,165,label)
    canvas.save()
    with fitz.open(path) as doc: doc[0].get_pixmap(matrix=fitz.Matrix(2.5,2.5)).save(path.with_suffix('.png'))
    base.dump(PUB/'curve.json',dict(labels=list(range(7)),metric='mean free glyph F2',series=series))


def feedback_example():
    grade=base.read(PUB/'phase.json')
    source=ARCHIVE
    inherited=ROOT/'publication/ilm-nature/evidence/breadth_pretrained_images.zip'
    phase=ROOT/'publication/ilm-nature/evidence/phase_feedback_images.zip'
    assert grade['archive']==base.sha(phase)
    i=next(j for j,g in enumerate(grade['grades']) if g['visual'] and not g['phase'])
    assets=FIG/'visual_training_breadth'
    items=[dict(type='heading',y=0,text='A change in reading back the same ink',meta='The Intelligent Investor')]
    y=15
    for archive,mode,label in ((source,'source','Source image'),(inherited,'pretrained','Pooled feedback'),(phase,'phase','Phase feedback')):
        member=f'{i:04d}/{mode}.png'
        with ZipFile(archive) as z:blob=z.read(member)
        name=member.replace('/','-')
        (assets/name).write_bytes(blob)
        with Image.open(BytesIO(blob)) as im:w,h=im.size
        sc=min(.47,385/w)
        items.append(dict(type='strip',x=99,y=y,w=w*sc,h=h*sc,label=label,name=name,member=member,
                          file=archive.name,sha256=sha256(blob).hexdigest()))
        y+=h*sc+3
    draw_plate(items,y,FIG/'Phase_feedback_example.pdf',assets,PUB/'phase-plate.json',phase)


def main():
    assets=FIG/'visual_training_breadth'
    assets.mkdir(exist_ok=True)
    s=base.read(PUB/'assessment.json')
    assert s['archive']==base.sha(ARCHIVE)
    items=[]
    y=0
    with ZipFile(ARCHIVE) as z, ZipFile(ROOT/'publication/ilm-nature/evidence/breadth_pretrained_images.zip') as alternate:
        rows=json.loads(z.read('records.json'))[:96]
        gain=next((i for i,g in enumerate(s['grades']) if g['broad'] and not g['narrow']),None)
        first=gain if gain is not None else next(i for i,r in enumerate(rows) if r['answers']['broad']['png']!=r['answers']['narrow']['png'])
        other=next(j%96 for j in range(first+1,first+97) if rows[j%96]['t']!=rows[first]['t'] and not s['grades'][j%96]['broad'])
        credited=next((i for i,g in enumerate(s['grades']) if rows[i]['t']==0 and g['broad'] and i!=first),None)
        examples=[] if credited is None else [(credited,'The first credited translation into English')]
        for i,heading in (examples+[(first,'A newly credited answer' if gain is not None else 'The first changed translation'),
                          (other,'An uncredited answer in the other direction')]):
            items.append(dict(type='heading',y=y,text=heading,meta=rows[i]['doc'].replace('-',' ').title()))
            y+=15
            for mode,label in (('source','Source image'),('narrow','Repeated corpus'),('broad','Broad corpus'),('pretrained','Pretrained reader')):
                member=f'{i:04d}/{mode}.png'
                blob=(alternate if mode=='pretrained' else z).read(member)
                fname=member.replace('/','-')
                (assets/fname).write_bytes(blob)
                with Image.open(BytesIO(blob)) as im:w,h=im.size
                scale=min(.53,385/w)
                items.append(dict(type='strip',x=99,y=y,w=w*scale,h=h*scale,label=label,name=fname,member=member,
                                  sha256=sha256(blob).hexdigest()))
                y+=max(14,h*scale)+2
            y+=6
    draw_plate(items,y,FIG/'Visual_training_breadth.pdf',assets,PUB/'plate.json',ARCHIVE)
    learning_curve()
    feedback_example()
    print('displayed',first,other,'gain',gain)


if __name__=='__main__':
    main()
