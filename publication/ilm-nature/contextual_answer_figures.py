"""Native S41 charts and fixed, unrepaired paired answer examples."""
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image
from generate_assignment_editable_figures import NS, Slide, chart_part, element, q, relpath, resolve, xml
from glyph_step_data import ROOT, HERE, EVIDENCE, load, require, sha

FIGURE = 'Figure_S41_contextual_image_answers'


def source_data():
    base = EVIDENCE/'glyph_tail_paired_answers_v2'; bound = {}
    def read(path):
        bound[str(path.relative_to(ROOT))] = sha(path)
        return load(path)
    archive = read(base/'archive.json')
    require(archive['total_streams']==144 and archive['both_conditions_retained'], 'retain both complete conditions')
    conditions = {}; cells = []
    for name in ['contextual','readout_only']:
        condition = archive['conditions'][name]
        for path,digest in condition['files_sha256'].items():
            actual = base/name/path
            require(sha(actual)==digest, 'archived paired evidence changed')
            bound[str(actual.relative_to(ROOT))] = digest
        scores = read(base/name/'scores.json'); position = read(base/name/'training_position_audit.json')
        audit = read(base/name/'final_checkpoint_audit.json')
        q = read(base/name/'qualification/report.json')
        require(not scores['pilot_complete_answer_requirement_passed'] and not scores['fixed_component_gate_passed'],
                'preserve both failed gates')
        conditions[name] = dict(groups=scores['groups'],paired_contexts=scores['fresh_both_questions_correct_contexts'],
            position=position,qualification=q,checkpoint_audit=audit)
        for index in range(8):
            row, = [r for r in scores['records'] if r['key']==f'fresh/{index:04}/normal']
            path = base/name/'raw_outputs'/row['png']
            require(sha(path)==row['png_sha256'] and Image.open(path).size==(max(1,row['glyphs'])*32,32),
                    'fixed actual output differs')
            cells.append(dict(condition=name,index=index,file=str(path.relative_to(ROOT)),sha256=sha(path),
                glyphs=row['glyphs'],stopped=row['stopped'],capped=row['capped'],
                displayed_glyphs=min(10,row['glyphs']),display_is_prefix_excerpt=row['glyphs']>10))
    cold = read(base/'cold_replay/verification.json')
    require(cold['actual_pixels_states_stopping_and_feedback_bit_exact'] and not cold['forbidden_paths']
            and not cold['IP_connection_attempts'], 'actual cold replay required')
    require([conditions[k]['groups']['fresh/normal']['manual_clear_complete_exact'] for k in conditions]==[16,11],
            'paired answer counts differ')
    return dict(conditions=conditions,cells=cells,source_files_sha256=bound,
        all144_outputs_preserved=True,fixed_example_indices=list(range(8)),new_balanced_fit_results_included=False)


def append(paper):
    require(paper.resolve()!=HERE.resolve(),'stage outside active publication')
    data=source_data()
    with zipfile.ZipFile(paper/'ILM_Figures_editable.pptx') as source:
        original={n:source.read(n) for n in source.namelist()}
    parts=dict(original);presentation=ET.fromstring(parts['ppt/presentation.xml'])
    ids=presentation.find('p:sldIdLst',NS)
    require(len(ids)==55 and sum(bool(re.fullmatch(r'ppt/charts/chart\d+\.xml',n)) for n in parts)==61,
            'stage S39 and S40 first')
    template='ppt/slides/slide55.xml';slide=Slide(parts[template]);trels=ET.fromstring(parts[relpath(template)])
    types=ET.fromstring(parts['[Content_Types].xml']);relationships=ET.fromstring(parts['ppt/_rels/presentation.xml.rels'])
    ct=next(raw for n,raw in parts.items() if re.fullmatch(r'ppt/charts/chart\d+\.xml',n)
        and ET.fromstring(raw).find('c:chart/c:plotArea/c:barChart',NS) is not None)
    charts=[];media=[]
    def chart(number,categories,values,series,maximum,unit,x):
        node,workbook=chart_part(ct,categories,values,series,0,maximum,'527A98')
        for item in node.findall('.//a:defRPr',NS):item.set('sz','1400')
        for axis in node.findall('c:chart/c:plotArea/c:valAx',NS):
            major=axis.find('c:majorUnit',NS)
            if major is None:major=element('c:majorUnit',axis)
            major.set('val',str(unit))
        part=f'ppt/charts/chart{number}.xml';book=f'ppt/embeddings/contextual_answer_{number}.xlsx'
        require(part not in parts and book not in parts,'chart collision')
        node.find('c:externalData',NS).set(q('r:id'),'rId1');rels=element('pkg:Relationships')
        element('pkg:Relationship',rels,Id='rId1',Type=NS['r']+'/package',Target='../embeddings/'+Path(book).name)
        parts[part],parts[book],parts[relpath(part)]=xml(node),workbook,xml(rels)
        element('ct:Override',types,PartName='/'+part,
            ContentType='application/vnd.openxmlformats-officedocument.drawingml.chart+xml')
        slide.chart(slide.relation('chart','../charts/'+Path(part).name),x,116,720,190)
        charts.append(dict(part=part,workbook=book,categories=categories,values=values,series=series,
            axis_range=[0,maximum],major_unit=unit))
    slide.shape('Contextual adaptation improves image answers; reliability remains limited',45,20,1510,50,26,True)
    slide.shape('a  Clear complete answers on new contexts',45,78,720,32,20,True)
    chart(62,['Adapted context','Readout only'],[16,11],'Clear complete matches / 32',32,8,45)
    slide.shape('Both questions correct: 5/16 vs 2/16 contexts.\nNeither meets the fixed 24/32 and 10/16 pilot.',45,309,720,60,14)
    slide.shape('b  Training first decision versus continuation',820,78,720,32,20,True)
    c=data['conditions']
    vals=[100*c[k]['position'][p]['mean_f1'] for k,p in
        [('contextual','first'),('contextual','later'),('readout_only','first'),('readout_only','later')]]
    chart(63,['Adapt first','Adapt later','Fixed first','Fixed later'],vals,'Foreground F1 (%)',100,25,820)
    slide.shape('First: 3,176; later: 20,086 reference-conditioned targets.\nThe first image is decided before any generated feedback.',820,309,720,60,14)
    slide.shape('c  Image prompt',45,382,250,43,17,True,center=True,fill='EAF1F6')
    slide.arrow(299,403,329,403)
    slide.shape('V + learned context',335,382,320,43,17,True,center=True,fill='EAF1F6')
    slide.arrow(659,403,689,403)
    slide.shape('Continuous G + image D',695,382,335,43,17,True,center=True,fill='EAF1F6')
    slide.arrow(1034,403,1064,403)
    slide.shape('Actual pixels -> E -> next state',1070,382,485,43,17,True,center=True,fill='EAF1F6')
    slide.shape('d  First eight reserved questions, both conditions, all outcomes retained',45,443,1510,32,19,True)
    lookup={(r['index'],r['condition']):r for r in data['cells']}
    for col in range(4):
        x=45+col*380
        for pair in range(2):
            index=col*2+pair;y=486+pair*178
            slide.shape(f'Question {index:02}',x,y,365,26,15,True)
            for j,name in enumerate(['contextual','readout_only']):
                row=lookup[index,name];top=y+31+j*64
                label=('Adapted' if name=='contextual' else 'Fixed')+f" | n={row['glyphs']} "+('cap' if row['capped'] else 'stop')
                if row['display_is_prefix_excerpt']:label+=' | first 10'
                slide.shape(label,x,top,365,23,12)
                path=ROOT/row['file'];part=f'ppt/media/contextual_answer_{name}_{path.name}'
                require(part not in parts,'media collision');parts[part]=path.read_bytes()
                slide.picture(slide.relation('image','../media/'+Path(part).name),path.stem+' actual pixels',
                    x+4,top+25,max(1,row['displayed_glyphs'])*24,24)
                if row['display_is_prefix_excerpt']:
                    fill=list(slide.tree)[-1].find('p:blipFill',NS)
                    fill.insert(1,element('a:srcRect',r=round(100000*(1-10/row['glyphs']))))
                media.append(dict(part=part,**row))
    slide.shape('Unrepaired output pixels. Prefixes only where marked; full PNGs embedded. Agent reading is not independent human grading. Gates remain failed.',
                45,866,1510,31,12)
    slide.relation('slideLayout',next(n.get('Target') for n in trels if n.get('Type').endswith('/slideLayout')))
    part='ppt/slides/slide56.xml';note='ppt/notesSlides/contextual_answer_notes56.xml'
    slide.relation('notesSlide','../notesSlides/'+Path(note).name)
    oldnote=next(resolve(template,n.get('Target')) for n in trels if n.get('Type').endswith('/notesSlide'))
    notes=ET.fromstring(parts[oldnote]);nrels=ET.fromstring(parts[relpath(oldnote)])
    caption=('Figure S41. Completed paired image-answer results. Clear literal matches are16/32 and11/32; '
        'paired contexts5/16 and2/16, canonical complete images2/32 and1/32. Both preset pilot and training-pixel gates fail. '
        'Training first/later scores are reference-conditioned, not autonomous accuracy. The first eight reserved questions '
        'in both conditions are fixed examples, including caps; full original output bytes are embedded. '
        'No new balanced-writer experiment result is included. The model retains a7.522B inherited body.')
    for shape in notes.findall('p:cSld/p:spTree/p:sp',NS):
        ph=shape.find('p:nvSpPr/p:nvPr/p:ph',NS)
        if ph is not None and ph.get('type')=='body':
            for i,node in enumerate(shape.findall('.//a:t',NS)):node.text=caption if i==0 else ''
    for node in nrels:
        if node.get('Type').endswith('/slide'):node.set('Target','../slides/slide56.xml')
    for name,kind in [(part,'slide'),(note,'notesSlide')]:
        require(name not in parts,'new slide/note collision')
        element('ct:Override',types,PartName='/'+name,
            ContentType='application/vnd.openxmlformats-officedocument.presentationml.'+kind+'+xml')
    parts[part],parts[relpath(part)]=xml(slide.root),xml(slide.rels)
    parts[note],parts[relpath(note)]=xml(notes),xml(nrels)
    rid=next(f'rId{i}' for i in range(1,1000) if f'rId{i}' not in {n.get('Id') for n in relationships})
    element('pkg:Relationship',relationships,Id=rid,Type=NS['r']+'/slide',Target='slides/slide56.xml')
    element('p:sldId',ids,id=max(int(n.get('id')) for n in ids)+1).set(q('r:id'),rid)
    mutable={'[Content_Types].xml','ppt/presentation.xml','ppt/_rels/presentation.xml.rels'}
    for name,node in [('[Content_Types].xml',types),('ppt/presentation.xml',presentation),
        ('ppt/_rels/presentation.xml.rels',relationships)]:parts[name]=xml(node)
    require(all(parts[n]==raw for n,raw in original.items() if n not in mutable),'predecessor part changed')
    with zipfile.ZipFile(paper/'ILM_Figures_editable.pptx','w',zipfile.ZIP_DEFLATED) as output:
        for name,raw in parts.items():output.writestr(name,raw)
    return dict(native_slides=56,native_charts=63,charts=charts,media=media,sources=data,
        prior55_slide_parts_preserved=True,prior61_chart_parts_preserved=True,runtime_outputs_repaired=False,
        subsequent_balanced_fit_results_included=False)
