"""Append Figure S39 as native OOXML with linked charts and unchanged raw cells."""
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from generate_assignment_editable_figures import NS,Slide,chart_part,element,q,relpath,resolve,xml
from motor_gate_data import HERE,ROOT,load,sha,require

FIGURE='Figure_S39_motor_uncertainty'
SOURCES=ROOT/'references/evidence/motor_gate_figure_sources_v1'


def source_data():
    data=load(SOURCES/'sources.json')
    require(data['indices']==[1,1362,2724,4086] and not data['images_repaired'],'fixed source grid differs')
    require(data['original_decoder_mixed_baselines_pixel_identical'],'unmatched baseline pixels')
    require(not data['genuine_curriculum_results_included'],'unfinished curriculum cannot enter this figure')
    for n,d in data['source_files_sha256'].items():require(sha(ROOT/n)==d,'completed source changed: '+n)
    for row in data['cells']:require(sha(SOURCES/row['file'])==row['sha256'],'raw glyph cell changed')
    return data


def append(paper):
    from PIL import Image
    require(paper.resolve()!=HERE.resolve(),'never overwrite the active deck during staging')
    data=source_data()
    with zipfile.ZipFile(HERE/'ILM_Figures_editable.pptx') as archive:
        original={n:archive.read(n) for n in archive.namelist()}
    parts=dict(original);template='ppt/slides/slide53.xml'
    presentation=ET.fromstring(parts['ppt/presentation.xml']);ids=presentation.find('p:sldIdLst',NS)
    require(len(ids)==53,'53-slide reviewed baseline required')
    require(len([n for n in parts if re.fullmatch(r'ppt/charts/chart\d+\.xml',n)])==57,'57 native charts required')
    types=ET.fromstring(parts['[Content_Types].xml'])
    relationships=ET.fromstring(parts['ppt/_rels/presentation.xml.rels'])
    template_rels=ET.fromstring(parts[relpath(template)])
    chart_template=next(raw for n,raw in parts.items() if re.fullmatch(r'ppt/charts/chart\d+\.xml',n)
        and ET.fromstring(raw).find('c:chart/c:plotArea/c:barChart',NS) is not None)
    charts=[];media=[];slide=Slide(parts[template])

    def chart(number,categories,values,series,maximum,unit,fmt,x,y,w,h):
        node,workbook=chart_part(chart_template,categories,values,series,0,maximum,'527A98')
        for f in node.findall('.//c:numFmt',NS):f.set('formatCode',fmt);f.set('sourceLinked','0')
        for f in node.findall('.//c:formatCode',NS):f.text=fmt
        for f in node.findall('.//a:defRPr',NS):f.set('sz','1500')
        for axis in node.findall('c:chart/c:plotArea/c:valAx',NS):
            major=axis.find('c:majorUnit',NS)
            if major is None:major=element('c:majorUnit',axis)
            major.set('val',str(unit))
        part=f'ppt/charts/chart{number}.xml';book=f'ppt/embeddings/motor_gate_{number}.xlsx'
        require(part not in parts and book not in parts,'native chart collision')
        node.find('c:externalData',NS).set(q('r:id'),'rId1')
        rels=element('pkg:Relationships')
        element('pkg:Relationship',rels,Id='rId1',Type=NS['r']+'/package',Target='../embeddings/'+Path(book).name)
        parts[part],parts[book],parts[relpath(part)]=xml(node),workbook,xml(rels)
        element('ct:Override',types,PartName='/'+part,ContentType='application/vnd.openxmlformats-officedocument.drawingml.chart+xml')
        slide.chart(slide.relation('chart','../charts/'+Path(part).name),x,y,w,h)
        charts.append(dict(part=part,workbook=book,categories=categories,values=values,series=series,
            axis_range=[0,maximum],major_unit=unit,number_format=fmt))

    def picture(row,x,y):
        path=SOURCES/row['file'];raw=path.read_bytes();im=Image.open(io.BytesIO(raw))
        require(im.size==(32,32),'complete glyph cell required')
        part='ppt/media/motor_gate_'+row['file'];require(part not in parts,'glyph media collision')
        parts[part]=raw
        slide.picture(slide.relation('image','../media/'+Path(part).name),row['column']+' '+str(row['index']),x,y,64,64)
        media.append(dict(part=part,source=str(path.relative_to(ROOT)),sha256=sha(path),original_size=[32,32],
            cell_crop=None,upstream_source=row['source'],upstream_sha256=row['source_sha256'],upstream_crop=row['crop']))

    slide.shape('Sharp known-form motors fail when semantic conditions mix glyphs',45,24,1510,62,28,True)
    slide.shape('a  Same 16 question-image initial states',45,105,710,40,21,True)
    a=data['first_state']
    chart(58,['Latent nearest','Image exact'],[a['normal_top_reference_form'],a['normal_first_image_exact']],
        'Allowed reference first forms / 16',16,4,'0',45,153,710,250)
    slide.shape('Reference-first-form diagnostics, not complete answer accuracy.\nNo nearest-form selection is used by the runtime; 5/16 first images reach F1 0.95.',
        55,410,700,62,15)
    slide.shape('b  Semantic output-span regression limits',820,105,735,40,21,True)
    r=data['regression']
    chart(59,['Trained span','Best span'],[r['trained_span_mse'],r['best_span_mse']],
        'Minimum uniform coordinate MSE',.35,.1,'0.00',820,153,735,250)
    slide.shape('4,087 anchors; 2,048-dimensional affine output.\nA latent regression bound, not a pixel or language-error bound.',835,410,710,62,15)
    slide.shape('c  Physical glyph generation on known forms',45,495,710,40,21,True)
    columns=[(45,330,'Motor'),(375,180,'Clean F1'),(555,180,'Mixed F1')]
    for x,w,label in columns:slide.shape(label,x,545,w,40,18,True,fill='EAF1F6')
    for i,row in enumerate(data['motor_tests']):
        y=590+i*58
        slide.shape(row['name'],45,y,330,48,18,fill='F0F3F5')
        slide.shape(f"{row['clean_f1']:.6f}",375,y,180,48,18,fill='F0F3F5')
        slide.shape(f"{row['mixed_f1']:.6f}",555,y,180,48,18,fill='F0F3F5')
    slide.shape('All rows assess 4,087 forms; training recipes and costs differ.\nEvery mixed-code gate fails; no autonomous QA follows these fits.',
        45,782,710,63,15)
    slide.shape('d  Actual 55/45-mixture outputs, unrepaired',820,495,735,40,21,True)
    names=['Target','D','Noise D','Dense flow','Spatial flow']
    for i,name in enumerate(names):slide.shape(name,845+i*135,539,125,34,16,True,center=True)
    for i,index in enumerate(data['indices']):
        y=580+i*70
        slide.shape(str(index),810,y+20,48,35,13)
        for j,name in enumerate(names):
            row=next(v for v in data['cells'] if v['index']==index and v['column']==name)
            picture(row,875+j*135,y)
    slide.shape('Fixed non-empty indices span the inventory. Complete 32 × 32 source pixels; no inferred or replacement strokes.',
        45,862,1510,35,14)

    slide.relation('slideLayout',next(n.get('Target') for n in template_rels if n.get('Type').endswith('/slideLayout')))
    part='ppt/slides/slide54.xml';note='ppt/notesSlides/motor_gate_notes54.xml'
    slide.relation('notesSlide','../notesSlides/'+Path(note).name)
    old=next(resolve(template,n.get('Target')) for n in template_rels if n.get('Type').endswith('/notesSlide'))
    notes=ET.fromstring(parts[old]);note_rels=ET.fromstring(parts[relpath(old)])
    caption=('Figure S39. Same16normal first-state diagnostics:9nearest reference forms,1exact first image. '
        'Minimum uniform semantic-code MSE compares the trained output span/bias with the best rank2048span; this is not a pixel bound. '
        'All4087known forms are evaluated for each separately trained motor. Every mixed gate fails. '
        'Four fixed non-empty indices1,1362,2724,4086show complete actual raw glyph cells under declared55/45conditions. '
        'No reference selects or repairs a neural output. Original-decoder mixed baseline pixels are identical across all three trials. '
        'No unfinished genuine-curriculum result is included; no language success is claimed for these motors.')
    for shape in notes.findall('p:cSld/p:spTree/p:sp',NS):
        ph=shape.find('p:nvSpPr/p:nvPr/p:ph',NS)
        if ph is not None and ph.get('type')=='body':
            for i,node in enumerate(shape.findall('.//a:t',NS)):node.text=caption if i==0 else ''
    for n in note_rels:
        if n.get('Type').endswith('/slide'):n.set('Target','../slides/slide54.xml')
    for name,kind in [(part,'slide'),(note,'notesSlide')]:
        require(name not in parts,'slide/note collision')
        element('ct:Override',types,PartName='/'+name,ContentType='application/vnd.openxmlformats-officedocument.presentationml.'+kind+'+xml')
    parts[part],parts[relpath(part)]=xml(slide.root),xml(slide.rels)
    parts[note],parts[relpath(note)]=xml(notes),xml(note_rels)
    used={n.get('Id') for n in relationships};rid=next(f'rId{i}' for i in range(1,1000) if f'rId{i}' not in used)
    element('pkg:Relationship',relationships,Id=rid,Type=NS['r']+'/slide',Target='slides/slide54.xml')
    element('p:sldId',ids,id=max(int(n.get('id')) for n in ids)+1).set(q('r:id'),rid)
    mutable={'[Content_Types].xml','ppt/presentation.xml','ppt/_rels/presentation.xml.rels'}
    for name,node in [('[Content_Types].xml',types),('ppt/presentation.xml',presentation),
                      ('ppt/_rels/presentation.xml.rels',relationships)]:parts[name]=xml(node)
    require(all(parts[n]==v for n,v in original.items() if n not in mutable),'prior OOXML part changed')
    with zipfile.ZipFile(paper/'ILM_Figures_editable.pptx','w',zipfile.ZIP_DEFLATED) as archive:
        for n,v in parts.items():archive.writestr(n,v)
    return dict(native_slides=54,native_charts=59,prior53_slide_parts_preserved=True,prior57_charts_preserved=True,
        charts=charts,media=media,source_manifest_sha256=sha(SOURCES/'sources.json'),genuine_curriculum_results_included=False)
