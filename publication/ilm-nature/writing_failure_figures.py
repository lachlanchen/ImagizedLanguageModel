"""Append native editable diagnostic figures; preserve all previous OOXML parts."""
import hashlib
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image
from generate_assignment_editable_figures import NS,Slide,chart_part,element,q,relpath,resolve,xml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIGURES=('Figure_S36_writing_failure','Figure_S37_semantic_glyph_loop','Figure_S38_date_interventions')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())


def append(paper):
    with zipfile.ZipFile(HERE/'ILM_Figures_editable.pptx') as archive:
        original={n:archive.read(n) for n in archive.namelist()}
    parts=dict(original);template='ppt/slides/slide50.xml'
    presentation=ET.fromstring(parts['ppt/presentation.xml']);ids=presentation.find('p:sldIdLst',NS)
    require(len(ids)==50,'50-slide baseline required')
    types=ET.fromstring(parts['[Content_Types].xml'])
    relationships=ET.fromstring(parts['ppt/_rels/presentation.xml.rels'])
    template_rels=ET.fromstring(parts[relpath(template)])
    template_chart=next(raw for name,raw in parts.items() if re.fullmatch(r'ppt/charts/chart\d+\.xml',name)
        and ET.fromstring(raw).find('c:chart/c:plotArea/c:barChart',NS) is not None)
    media=[];chart_data=[]

    def register(slide,number,caption):
        layout=next(n.get('Target') for n in template_rels if n.get('Type').endswith('/slideLayout'))
        slide.relation('slideLayout',layout)
        part=f'ppt/slides/slide{number}.xml';note=f'ppt/notesSlides/writing_failure_notes{number}.xml'
        slide.relation('notesSlide','../notesSlides/'+Path(note).name)
        old=next(resolve(template,n.get('Target')) for n in template_rels if n.get('Type').endswith('/notesSlide'))
        notes=ET.fromstring(parts[old]);note_rels=ET.fromstring(parts[relpath(old)])
        for shape in notes.findall('p:cSld/p:spTree/p:sp',NS):
            placeholder=shape.find('p:nvSpPr/p:nvPr/p:ph',NS)
            if placeholder is not None and placeholder.get('type')=='body':
                for i,node in enumerate(shape.findall('.//a:t',NS)):node.text=caption if i==0 else ''
        for node in note_rels:
            if node.get('Type').endswith('/slide'):node.set('Target',f'../slides/slide{number}.xml')
        for name,kind in [(part,'slide'),(note,'notesSlide')]:
            require(name not in parts,'slide collision')
            element('ct:Override',types,PartName='/'+name,
                ContentType='application/vnd.openxmlformats-officedocument.presentationml.'+kind+'+xml')
        parts[part],parts[relpath(part)]=xml(slide.root),xml(slide.rels)
        parts[note],parts[relpath(note)]=xml(notes),xml(note_rels)
        used={n.get('Id') for n in relationships};rid=next(f'rId{i}' for i in range(1,1000) if f'rId{i}' not in used)
        element('pkg:Relationship',relationships,Id=rid,Type=NS['r']+'/slide',Target=f'slides/slide{number}.xml')
        element('p:sldId',ids,id=max(int(n.get('id')) for n in ids)+1).set(q('r:id'),rid)

    def chart(slide,number,categories,values,title,minimum,maximum,x,y,w,h):
        node,workbook=chart_part(template_chart,categories,values,title,minimum,maximum,'527A98')
        # Integer counts and nats are not displayed as spurious three-decimal counts.
        for fmt in node.findall('.//c:numFmt',NS):fmt.set('formatCode','0');fmt.set('sourceLinked','0')
        for props in node.findall('.//a:defRPr',NS):props.set('sz','1500')
        name=f'ppt/charts/chart{number}.xml';book=f'ppt/embeddings/writing_failure_{number}.xlsx'
        require(name not in parts and book not in parts,'chart collision')
        node.find('c:externalData',NS).set(q('r:id'),'rId1')
        rels=element('pkg:Relationships')
        element('pkg:Relationship',rels,Id='rId1',Type=NS['r']+'/package',Target='../embeddings/'+Path(book).name)
        parts[name],parts[book],parts[relpath(name)]=xml(node),workbook,xml(rels)
        element('ct:Override',types,PartName='/'+name,
            ContentType='application/vnd.openxmlformats-officedocument.drawingml.chart+xml')
        slide.chart(slide.relation('chart','../charts/'+Path(name).name),x,y,w,h)
        chart_data.append(dict(part=name,workbook=book,categories=categories,values=values,series=title))

    def picture(slide,path,name,x,y,w):
        raw=path.read_bytes();im=Image.open(io.BytesIO(raw))
        part='ppt/media/writing_failure_'+name+'.png';require(part not in parts,'media collision')
        parts[part]=raw
        slide.picture(slide.relation('image','../media/'+Path(part).name),name,x,y,w,w*im.height/im.width)
        media.append(dict(part=part,source=str(path.relative_to(ROOT)),sha256=sha(path),
                          original_size=list(im.size),crop=None))

    s=Slide(parts[template]);s.shape('The reader retains answers that the learned pixel writer loses',45,24,1510,60,28,True)
    s.shape('a  Complete pixel-exact training answers / 64',45,105,730,40,21,True)
    chart(s,55,['0','4,000','8,000','16,000'],[0,3,21,60],'Same 64 training questions',0,64,45,153,710,250)
    s.shape('Final full training: 495/512 exact.\nNormal development: 0/16 Chinese, 0/8 English.\nAll 512 training first images exact; all 24 normal development first images inexact.',
            60,405,695,92,17)
    s.shape('b  Correct first image loses likelihood',820,105,735,40,21,True)
    values=[]
    for step in [0,4000,8000,16000]:
        report=load(ROOT/f'references/evidence/pixel_language_first_block_likelihood/scores_{step:05}.json')
        rows=[report[f'development_{i:04}_normal'] for i in range(16)]
        values.append(sum(r['maximum_reference_minus_greedy'] for r in rows)/16)
    require(abs(values[-1]+751.243727)<1e-5,'likelihood source differs')
    chart(s,56,['0','4,000','8,000','16,000'],values,'Best reference minus emitted image, nats',-2800,0,820,153,735,250)
    s.shape('Final Chinese mean: -751.24 nats. Every reference loses.\nRelative preferences still transfer: 14/16 favor the owning question; all 8 paired assignments positive.',
            835,405,710,88,17)
    s.shape('c  Native reader: literal text matches / 16',45,525,730,40,21,True)
    chart(s,57,['Normal image','Question erased','Source text'],[12,0,11],'Helper-only native reader',0,16,45,573,710,225)
    s.shape('Same 32 image-memory fingerprints as the ILM.\nA symbolic diagnostic baseline, not an image-output solution.',60,800,700,57,16)
    s.shape('d  Actual pixel outputs remain damaged',820,525,735,40,21,True)
    base=ROOT/'references/evidence/pixel_language_diversity_final16000/raw_outputs'
    for y,index,answer in [(585,12,'1874'),(690,15,'2001')]:
        s.shape('Question '+str(index+1)+'  |  helper year '+answer,835,y,690,35,18,True)
        path=base/f'development_{index:04}_normal.png'
        picture(s,path,'date_'+str(index),835,y+40,Image.open(path).width*1.5)
    s.shape('Unrepaired final outputs. No missing stroke is inferred toward gold.\nLiteral text, image exactness and relative preference are different measurements.',
            835,805,700,60,16)
    register(s,51,'Figure S36. Broader512-question training fits495 complete answers but fails all24normal development answers. '
        'Common64 training count, normalized first-image likelihood margins and helper native-reader literal counts are distinct metrics. '
        'All32 helper image fingerprints match the ILM. Actual date images are retained without pixel changes. '
        'The helper is not an ILM image answer. Sources: pixel_language_diversity_final16000, pixel_language_first_block_likelihood, pixel_language_source_reader.')

    s=Slide(parts[template]);s.shape('A complete-glyph semantic loop separates writing from continuation',45,24,1510,60,28,True)
    s.shape('a  Evaluated language runtime: one whole glyph per event',45,105,1510,40,21,True)
    for x,w,text in [(45,240,'Question image\nNative visual reader'),(345,280,'Original assistant state h\nFull native context'),
                     (685,245,'Continuous readout G\nSemantic vector z'),(990,245,'Neural decoder D\n32 × 32 glyph pixels'),
                     (1295,260,'Image encoder E\nActual emitted pixels')]:
        s.shape(text,x,160,w,115,18,center=True,fill='EAF1F6' if x in (345,685) else 'F0F3F5')
    for x1,x2 in [(285,340),(625,680),(930,985),(1235,1290)]:s.arrow(x1,217,x2,217)
    s.arrow(1425,280,1425,320,head=False);s.arrow(1425,320,480,320,head=False);s.arrow(480,320,480,280)
    s.shape('Feedback uses generated pixels. Learned stop; no reference answer length.',570,285,825,32,16)
    s.shape('b  Completed primitive training',45,365,730,40,21,True)
    s.shape('Offline source anchors + canonical glyph images\nPrimitive: train E/D; freeze readout/stopping.\nLater: fit readout/stopping on 508 question-image histories.\n4,096 anchored characters → 4,087 distinct image forms.',
            45,415,730,135,18)
    s.shape('Direct: 4,048/4,087 exact; F1 0.999940\nRoundtrip: 3,697/4,087 exact; F1 0.999634\nSemantic cosine: 0.77735 on first 128; alignment remains weaker.\n80.22 s / 0.728 GiB for the primitive only.',
            820,415,735,135,18)
    s.shape('c  Gold / direct decode / actual-image roundtrip',45,560,1510,35,21,True)
    picture(s,ROOT/'references/evidence/semantic_glyph_bridge/glyph_reconstruction_board.png',
            'glyph_triplets',60,603,1480)
    s.shape('64 evenly spaced indices; all physical pixels retained. This is reconstruction, not answer accuracy.\nBroader Chinese assessment: 1/16 exact, 11 caps. No runtime tokenizer, source text head, font renderer or glyph lookup.',
            45,850,1510,45,14)
    register(s,52,'Figure S37. Continuous complete-glyph loop and completed primitive. '
        'The native assistant state is preserved; actual emitted images determine feedback. '
        'Primitive E/D results are separate from the later 508-question readout fit. The 64 image triplets use evenly spaced inventory indices, without quality selection. '
        'Mean cosine is reported on the first 128 prepared forms. Broader normal Chinese assessment has 1/16 pixel-exact and 11 capped responses. Unknown glyphs and historical scripts remain unvalidated.')

    s=Slide(parts[template]);s.shape('Changed passage dates produce changed answer images',45,24,1510,60,28,True)
    s.shape('a  Frozen model, two selected question templates, zero further updates',45,105,1510,40,21,True)
    s.shape('Replace the passage year\n1874/2001 → fixed counterfactual year\nKeep original geometry and question ink',45,155,680,112,19,
            center=True,fill='F0F3F5')
    s.arrow(735,208,810,208)
    s.shape('Read the complete image and emit glyph pixels\nSame 64-glyph cap and learned stopping\nErased-question controls use the same passage',820,155,735,112,19,
            center=True,fill='EAF1F6')
    s.shape('b  All normal outputs in fixed order. Labels state requested years; images are actual model output.',45,295,1510,40,19,True)
    evidence=ROOT/'references/evidence/semantic_glyph_counterfactual'
    normal=[r for r in load(evidence/'report.json')['records'] if r['route']=='normal']
    require(len(normal)==16,'all 16 normal date outputs required')
    for i,row in enumerate(normal):
        x=45+(i%4)*385;y=350+(i//4)*119
        label=('Treaty' if row['parent']==12 else 'Coast')+' | requested '+row['year']
        s.shape(label,x,y,355,35,18,True)
        path=evidence/'raw_outputs'/row['png']
        require(Image.open(path).size==(160,32),'complete five-glyph image required')
        picture(s,path,'counter_'+str(i),x+8,y+41,320)
    score=load(evidence/'review_scores.json')['groups']['normal_new_dates']
    require((score['cases'],score['canonical_pixel_exact'],score['output_first_manual_transcription_matches'])==(14,2,11),
            'date score source differs')
    s.shape('New dates: 11/14 clear output-first readings, 2/14 pixel-exact, 14/14 natural stops. Three damaged/wrong digit forms remain failures.\nTwo parents were chosen from prior successes; no sealed or independent semantic accuracy claim. All 32 raw outputs and controls retained.',
            45,838,1510,59,15)
    register(s,53,'Figure S38. Sixteen normal date outputs in fixed preparation order, including two original-input cold reruns. '
        'Labels indicate requested years and do not transcribe or repair output. Both 1789 variants and the coast 1648 variant have uncertain/wrong digits and are not counted as correct. '
        'Eleven of 14 new conditions have clear matching output-first manual transcriptions; two are canonical pixel-exact; all 14 stop without target length. '
        'Six clear changed-date outputs compose complete strings absent from supported training answers. All erased controls fail date answering. '
        'Two parents are selected from previously observed successes; no independent human grading or sealed test. Source: references/evidence/semantic_glyph_counterfactual/.')

    for name,node in [('[Content_Types].xml',types),('ppt/presentation.xml',presentation),
                      ('ppt/_rels/presentation.xml.rels',relationships)]:parts[name]=xml(node)
    mutable={'[Content_Types].xml','ppt/presentation.xml','ppt/_rels/presentation.xml.rels'}
    require(all(parts[n]==raw for n,raw in original.items() if n not in mutable),'old package part changed')
    with zipfile.ZipFile(paper/'ILM_Figures_editable.pptx','w',zipfile.ZIP_DEFLATED) as archive:
        for n,raw in parts.items():archive.writestr(n,raw)
    return dict(native_slides=53,native_charts=57,prior50_slide_parts_preserved=True,
                prior54_charts_preserved=True,charts=chart_data,media=media)
