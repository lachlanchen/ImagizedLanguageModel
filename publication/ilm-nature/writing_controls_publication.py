#!/usr/bin/env python3
"""Append source-bound writing controls without rewriting prior native slides."""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import fitz
import openpyxl
from PIL import Image, ImageChops

from generate_assignment_editable_figures import NS, Slide, chart_part, element, q, relpath, resolve, xml
from native_flow_manuscript import (DOCS, HERE, ROOT, TOOL, TOOL_SHA, build_slot,
    expanded, owned_command, protected, require, save, sha)
from render_native_qa_publication import pdf_pixels

BASE='3011de62448a3228c3605439ed1f663169a4ff77'
FIGURES=('Figure_S32_native_writing_controls','Figure_S33_conditional_pixel_fitting')
EVIDENCE=ROOT/'references/evidence'
SHORT=EVIDENCE/'conditional_pixel_density_result'
LONG=EVIDENCE/'pixel_density_long_fit_result'
PREFIX=EVIDENCE/'pixel_prefix_diagnosis_result'
WRITER=EVIDENCE/'native_writer_content_control_result'
ABLATION=EVIDENCE/'native_reference_latent_ablation_result'
QUERY=EVIDENCE/'native_query_fit_result'


def load(path):
    return json.loads(Path(path).read_text())


def baseline(name):
    return subprocess.check_output(['git','show',BASE+':publication/ilm-nature/'+name],cwd=ROOT)


def source_evidence():
    bound={}
    for folder,key in [(SHORT,'archive_files_sha256'),(LONG,'archive_files_sha256')]:
        record=load(folder/'result.json')
        for name,digest in record[key].items():
            require(sha(folder/name)==digest,'source archive changed: '+name)
    for folder in (QUERY,WRITER,ABLATION):
        record=load(folder/'provenance.json')
        for name,digest in record['files_sha256'].items():
            require(sha(folder/name)==digest,'source native control changed: '+name)
    for folder in (SHORT,LONG,PREFIX,QUERY,WRITER,ABLATION):
        for path in sorted(folder.rglob('*')):
            if path.is_file():bound[str(path.relative_to(ROOT))]=sha(path)
    short,long=load(SHORT/'audit.json'),load(LONG/'audit.json')
    require(long['groups']['train_cmrc_normal']['exact']==256
        and long['groups']['train_dolly_normal']['exact']==256
        and long['groups']['development_cmrc_normal']['exact']==0
        and long['groups']['development_dolly_normal']['exact']==0,
        'completed independently scored fitting/generalization result required')
    require(load(LONG/'raw_review.json')['all_six_plates_actually_displayed'], 'complete long-fit raw review required')
    return bound,short,long


def append_figures(paper,stage):
    with zipfile.ZipFile(HERE/'ILM_Figures_editable.pptx') as archive:
        original={name:archive.read(name) for name in archive.namelist()}
    parts=dict(original)
    presentation=ET.fromstring(parts['ppt/presentation.xml'])
    ids=presentation.find('p:sldIdLst',NS)
    require(len(ids)==46,'reviewed46-slide predecessor required')
    types=ET.fromstring(parts['[Content_Types].xml'])
    relationships=ET.fromstring(parts['ppt/_rels/presentation.xml.rels'])
    template='ppt/slides/slide46.xml'
    template_rels=ET.fromstring(parts[relpath(template)])
    media=[]

    def picture(slide,path,name,x,y,width,maximum_height=None,crop=False):
        raw=path.read_bytes()
        with Image.open(io.BytesIO(raw)) as opened:
            image=opened.convert('RGB')
        box=None
        if crop:
            ink=image.convert('L').point(lambda value:255 if value<224 else 0).getbbox()
            if ink:
                box=(max(0,ink[0]-8),max(0,ink[1]-8),min(image.width,ink[2]+8),min(image.height,ink[3]+8))
                image=image.crop(box)
                stream=io.BytesIO();image.save(stream,format='PNG');raw=stream.getvalue()
        w,h=width,width*image.height/image.width
        if maximum_height is not None and h>maximum_height:
            w*=maximum_height/h;h=maximum_height
        part='ppt/media/writing_controls_'+name+'.png'
        require(part not in parts,'media collision')
        parts[part]=raw
        slide.picture(slide.relation('image','../media/'+Path(part).name),name,x,y,w,h)
        media.append(dict(part=part,source=str(path.relative_to(ROOT)),source_sha256=sha(path),
            embedded_sha256=hashlib.sha256(raw).hexdigest(),crop_box=box,
            display_crop_luminance_bound=224 if crop else None,
            RGB_size=list(image.size),RGB_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest()))

    def table(slide,rows,x,y,widths,line=31,size=18):
        for i,row in enumerate(rows):
            left=x
            for j,value in enumerate(row):
                slide.shape(str(value),left,y+i*line,widths[j],line,size,bold=i==0,center=j>0)
                left+=widths[j]

    def register(slide,number,caption):
        layout=next(n.get('Target') for n in template_rels if n.get('Type').endswith('/slideLayout'))
        slide.relation('slideLayout',layout)
        part=f'ppt/slides/slide{number}.xml'
        note=f'ppt/notesSlides/writing_controls_notes{number}.xml'
        slide.relation('notesSlide','../notesSlides/'+Path(note).name)
        old_note=next(resolve(template,n.get('Target')) for n in template_rels if n.get('Type').endswith('/notesSlide'))
        notes=ET.fromstring(parts[old_note]);note_rels=ET.fromstring(parts[relpath(old_note)])
        for shape in notes.findall('p:cSld/p:spTree/p:sp',NS):
            placeholder=shape.find('p:nvSpPr/p:nvPr/p:ph',NS)
            if placeholder is not None and placeholder.get('type')=='body':
                for i,t in enumerate(shape.findall('.//a:t',NS)):t.text=caption if i==0 else ''
        for n in note_rels:
            if n.get('Type').endswith('/slide'):n.set('Target',f'../slides/slide{number}.xml')
        for name,kind in [(part,'slide'),(note,'notesSlide')]:
            require(name not in parts,'new slide/notes collision')
            element('ct:Override',types,PartName='/'+name,
                ContentType='application/vnd.openxmlformats-officedocument.presentationml.'+kind+'+xml')
        parts[part],parts[relpath(part)]=xml(slide.root),xml(slide.rels)
        parts[note],parts[relpath(note)]=xml(notes),xml(note_rels)
        used={n.get('Id') for n in relationships}
        rid=next(f'rId{i}' for i in range(1,1000) if f'rId{i}' not in used)
        element('pkg:Relationship',relationships,Id=rid,Type=NS['r']+'/slide',Target=f'slides/slide{number}.xml')
        element('p:sldId',ids,id=max(int(n.get('id')) for n in ids)+1).set(q('r:id'),rid)

    slide=Slide(parts[template])
    slide.shape('Native reader and writer controls retain inaccurate answers',45,24,1510,60,30,True)
    slide.shape('a  Continuous reader-query fit',45,110,690,40,22,True)
    slide.shape('128 real updates of 262,144 shared query coordinates\n6.240B retained native flow model, image-only input',45,158,700,65,18)
    table(slide,[['Canonical OCR exact','Before','After'],['Seen questions','0/2','0/2'],
        ['Held-out questions','0/2','0/2'],['Seen / held-out copies','0/4','0/4'],
        ['Clean reference pages','8/8','8/8']],45,238,[390,140,140])
    slide.shape('b  Correct answer text supplied to the native writer',795,110,760,40,22,True)
    slide.shape('Fixed four strings, guidance 4\n6.863B retained diagnostic runtime, including lexical modules',795,158,760,65,18)
    table(slide,[['OCR exact','Full reference','Reference removed'],['Source Chinese','0/2','0/2'],
        ['Source English','1/2','0/2'],['Adapted, combined','0/4','0/4'],['Clean references','4/4','4/4']],
        795,238,[340,180,235])
    slide.shape('c  Every supplied-content case, source writer',45,432,1510,35,22,True)
    slide.shape('Target',45,522,160,30,18,True)
    slide.shape('Full reference',45,627,180,30,18,True)
    slide.shape('Reference removed',45,732,180,30,18,True)
    supplied=[r for r in load(WRITER/'assessment.json')['rows']
        if r['arm']=='source' and r['kind']=='supplied_content' and r['guidance']==4.]
    removed={r['id']:r for r in load(ABLATION/'assessment.json')['rows']
        if r['arm']=='source' and r['kind']=='reference_removed'}
    require(len(supplied)==4 and len(removed)==4,'all four supplied-content cases required')
    for i,row in enumerate(supplied):
        x=260+320*i
        slide.shape(('Chinese ' if i<2 else 'English ')+str(i%2+1),x,477,300,32,18,True)
        picture(slide,WRITER/'references'/f'{i:02}.png',f'target_{i}',x,520,290,83,True)
        for y,folder,path,kind in [(625,WRITER/'images',row['path'],'full'),
                (730,ABLATION/'images',removed[row['id']]['path'],'removed')]:
            picture(slide,folder/path,f'{kind}_page_{i}',x,y,38,83,False)
            picture(slide,folder/path,f'{kind}_ink_{i}',x+48,y,242,83,True)
    slide.shape('All four cases in original order. Full-page thumbnails accompany fixed ink crops (display threshold 224). Pixel values and scores are unchanged. Supplied text is an offline control.',45,844,1510,48,15)
    register(slide,47,'Figure S32. Completed reader-query fit and native supplied-content/reference-removal controls. '
        'All query-fit answer counts remain zero; clean reference OCR is 8/8. Source supplied-content writing is 1/4 with the full reference and 0/4 after removal; adapted writing remains 0/4. '
        'The one successful source English page is included among all four fixed cases. Full-page thumbnails accompany enlarged crops selected by luminance below224 with margin8. Cropped pixel values are unchanged; complete original hashes and crop boxes are bound in preparation. '
        'These reused small controls do not identify a unique repair or establish autonomous language understanding. '
        'Sources: references/evidence/native_query_fit_result/assessment.json, native_writer_content_control_result/assessment.json, native_reference_latent_ablation_result/assessment.json.')

    slide=Slide(parts[template])
    slide.shape('Direct pixel likelihood fits known blocks but fails held-out writing',45,24,1510,60,29,True)
    slide.shape('a  Conditional physical-image density',45,108,1510,35,22,True)
    for x,w,text in [(45,280,'Cached state h\n2 × 4096\nFrozen V68 actor'),
        (405,350,'Causal pixel decoder\n4 layers, 3.453M learned parameters'),
        (835,300,'256 eight-pixel patterns\n2 × 4 physical pieces'),(1215,340,'32 × 64 emitted image\nExact inverse packing')]:
        slide.shape(text,x,155,w,100,18,center=True,fill='EAF1F6' if x==405 else 'F0F3F5')
    for a,b in [(330,400),(760,830),(1140,1210)]:slide.arrow(a,205,b,205)
    slide.shape('Training supplies preceding target pieces. Generation uses only its own emitted pieces. No lexical output inventory or font renderer.',45,270,1510,40,17)
    slide.shape('b  Exact autonomous first blocks',45,325,640,35,21,True)
    table(slide,[['Cohort / updates','1,200','16,000'],['CMRC training','0/256','256/256'],
        ['Dolly training','12/256','256/256'],['CMRC development','0/128','0/128'],['Dolly development','0/64','0/64']],
        45,370,[300,170,170],line=30,size=17)
    slide.shape('Whole-training-set NLL per eight-pixel piece (log scale)',45,540,640,35,18,True)
    monitor=load(LONG/'monitor.json');categories=[str(r['update']) for r in monitor]
    template_chart=next(raw for name,raw in parts.items() if re.fullmatch(r'ppt/charts/chart\d+\.xml',name)
        and ET.fromstring(raw).find('c:chart/c:plotArea/c:barChart',NS) is not None)
    chart,workbook=chart_part(template_chart,categories,[r['train_zh_nll'] for r in monitor],
        'CMRC',1e-9,10,'527A98')
    second,_=chart_part(template_chart,categories,[r['train_en_nll'] for r in monitor],
        'Dolly',1e-9,10,'89979F')
    bar=chart.find('c:chart/c:plotArea/c:barChart',NS)
    ser=copy.deepcopy(second.find('c:chart/c:plotArea/c:barChart/c:ser',NS))
    ser.find('c:idx',NS).set('val','1');ser.find('c:order',NS).set('val','1')
    for f in ser.findall('.//c:f',NS):f.text=f.text.replace('$B$','$C$')
    bar.insert(list(bar).index(bar.find('c:ser',NS))+1,ser)
    labels=bar.find('c:dLbls',NS)
    if labels is not None:bar.remove(labels)
    for axis in chart.findall('c:chart/c:plotArea/c:valAx',NS):
        element('c:logBase',axis.find('c:scaling',NS),val='10')
        unit=axis.find('c:majorUnit',NS)
        if unit is not None:axis.remove(unit)
        fmt=axis.find('c:numFmt',NS)
        if fmt is not None:fmt.set('formatCode','0E+00');fmt.set('sourceLinked','0')
    for run in chart.findall('.//a:defRPr',NS):run.set('sz','1500')
    wb=openpyxl.Workbook();sheet=wb.active;sheet.title='Sheet1';sheet.append(['Update','CMRC','Dolly'])
    for row in monitor:sheet.append([str(row['update']),row['train_zh_nll'],row['train_en_nll']])
    buffer=io.BytesIO();wb.save(buffer);workbook=buffer.getvalue()
    chartname='ppt/charts/chart53.xml';bookname='ppt/embeddings/writing_controls_NLL.xlsx'
    require(chartname not in parts and bookname not in parts,'new native chart collision')
    parts[chartname]=xml(chart);parts[bookname]=workbook
    rels=element('pkg:Relationships');element('pkg:Relationship',rels,Id='rId1',
        Type=NS['r']+'/package',Target='../embeddings/'+Path(bookname).name)
    parts[relpath(chartname)]=xml(rels)
    external=chart.find('c:externalData',NS)
    if external is not None:external.set(q('r:id'),'rId1');parts[chartname]=xml(chart)
    element('ct:Override',types,PartName='/'+chartname,
        ContentType='application/vnd.openxmlformats-officedocument.drawingml.chart+xml')
    slide.chart(slide.relation('chart','../charts/chart53.xml'),45,585,640,220)
    slide.shape('CMRC (blue)    Dolly (grey)       Optimizer updates',90,804,590,29,15)
    slide.shape('c  Reference, shorter fit and longer fit',745,325,810,35,21,True)
    indices=(1,256,512,640)
    labels=('CMRC train (1)','Dolly train (256)','CMRC dev (512)','Dolly dev (640)')
    for j,(index,label) in enumerate(zip(indices,labels)):
        x=850+180*j
        slide.shape(label,x-15,375,180,35,16,True)
        for y,folder,name in [(430,PREFIX/'images',f'{index:04}_reference.png'),
                (540,SHORT/'images',f'{index:04}.png'),(650,LONG/'images',f'{index:04}.png')]:
            if index==1 and y==430:
                # This is an original target, rendered only during dataset preparation.
                import torch
                data=torch.load(ROOT/'artifacts/pixel_density_long_fit_v1/data.pt',map_location='cpu',weights_only=True)
                path=stage/'reference_0001.png'
                Image.fromarray((~data['ink'][1].numpy()).astype('uint8')*255).save(path)
            else:path=folder/name
            picture(slide,path,f'block_{index}_{y}',x,y,144,72)
    for y,text in [(450,'Reference'),(560,'1,200'),(670,'16,000')]:slide.shape(text,725,y,125,30,17,True)
    slide.shape('Original 32 × 64 physical blocks. Boundary-clipped glyphs are retained.',745,760,810,48,16)
    slide.shape('Longer fit: 4.65 min, 0.825 GiB head peak. Costs exclude the pretrained reader and cache. All 896 outputs scored; training memorization does not establish language generalization.',45,844,1510,48,15)
    register(slide,48,'Figure S33. Direct normalized physical-pixel likelihood, complete fitting counts and fixed examples. '
        'The V68 custom image interface supplies cached two-by4096 states; no reader is loaded during head fitting. The pixel alphabet contains physical eight-bit patterns, not word identities. '
        'The longer control restores the same untrained initialization and first75 permutations; learning-rate decay also spans the longer schedule. All512 training first blocks become exact, while all192 normal development blocks remain inexact. '
        'The native chart uses matched full-training-set NLL measurements, not unmatched minibatch losses. Its logarithmic axis reports nats per eight-pixel piece. '
        'Fixed examples are the first CMRC training reference with Chinese forms (1), first Dolly training (256), first CMRC development (512), first Dolly development (640). The first CMRC reference (0) is a Latin name. '
        'These are first blocks, not full answers. No independent semantic accuracy, historical-script qualification or useful-quality efficiency claim. '
        'Sources: references/evidence/conditional_pixel_density_result/audit.json, pixel_prefix_diagnosis_result/report.json, pixel_density_long_fit_result/audit.json and monitor.json.')
    for name,node in [('[Content_Types].xml',types),('ppt/presentation.xml',presentation),
            ('ppt/_rels/presentation.xml.rels',relationships)]:parts[name]=xml(node)
    mutable={'[Content_Types].xml','ppt/presentation.xml','ppt/_rels/presentation.xml.rels'}
    require(all(parts[name]==raw for name,raw in original.items() if name not in mutable),'prior native package part changed')
    require(sum(bool(re.fullmatch(r'ppt/charts/chart\d+\.xml',n)) for n in parts)==53,'53 native charts required')
    with zipfile.ZipFile(paper/'ILM_Figures_editable.pptx','w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in parts.items():archive.writestr(name,raw)
    return dict(media=media,prior46_slide_parts_preserved=True,prior52_chart_parts_preserved=True,
        native_slides=48,native_charts=53,artifact_tool_available=False,
        method='existing native ZIP/XML append, preserving source package parts')


def tex_fragments(paper):
    (paper/'writing_controls_results.tex').write_text(r'''Additional controls separate optimization from useful answering
(Supplementary Notes 54--56; Figs.~S32 and S33). A 128-update reader-query fit
still gives no exact answers on two seen and two excluded questions.
Supplying correct text to the native writer yields 1/4 source pages at
guidance four, falling to 0/4 when the VAE reference is removed; adapted
writing remains 0/4. A separate 3.453M-parameter causal pixel head learns
from frozen image-only contextual states. After 1,200 updates, even correct
preceding pixels leave Chinese training-piece accuracy at 76.831\%.
Restoring the same untrained head and extending optimization to 16,000
updates gives 512/512 exact autonomous training first blocks, but 0/128
Chinese and 0/64 English development first blocks. Active head training
takes 279.131\,s at 0.825\,GiB peak allocation, excluding the pretrained
reader and cache. Thus precise emission for learned conditions is achievable
in this component; conditional generalization and complete language answering
remain unresolved. Physical fitting is not a semantic benchmark.
''')
    (paper/'writing_controls_notes.tex').write_text(r'''\clearpage
\section*{Supplementary Note 54 | Continuous reader-query fitting}
\label{note:query-fit}
The native flow model of Note~53 receives 128 AdamW updates to 262,144 shared
continuous reader-query coordinates. Other weights, including the completed
connector/writer adapters, remain frozen. The retained system has
6,240,279,775 parameters. Sixteen updates use independent full noise and 112
use the other registered noise levels. Fifty-step guidance-one inference
uses image-only questions, without an OCR training loss or changing text
condition. Both seen questions, both excluded questions and four copying
cases remain OCR-inexact before and after fitting. Both new-seed seen answers
and both question-/passage-erased seen answers also fail. All eight unique
clean reference pages pass the same fixed OCR rule.

Every FP32 query coordinate changes; 251,820 also change after BF16
conversion. Saved optimizer counters, gradients, frozen-weight versions and
two-update replay checks pass. Seen velocity MSE changes from 0.003543 to
0.003429 and from 0.022216 to 0.019156. Unbounded normalized question-field
alignment changes from 0.074015 to 0.146650 on one passage. This is physical
optimization, not accuracy. The 18 new and four reused baseline pages are
authenticated and actually inspected. Whole execution takes 1,447.389\,s at
15.361\,GiB peak allocated memory. No useful answering or unique failure
mechanism is established by these small reused cohorts.

\section*{Supplementary Note 55 | Supplied content and reference removal}
\label{note:writer-controls}
Four fixed strings contain the two seen Chinese answers and one seen/one
excluded English copying string. Each native generation receives the complete
original input page and correct answer text through the original tokenizer
and lexical table. Targets never enter generation. This is an offline
supplied-content diagnostic, not independent image-language inference.
Source and completed 1,536-update adapted arms retain original reader queries;
the later failed query-fit checkpoint is not loaded. The diagnostic retains
6,862,609,631 pretrained parameters including lexical modules and an unused
output head, distinct from the vocabulary-removed system in Note~53.

\begin{center}\begin{tabular}{lrrr}
Writer / guidance & Chinese & English & Combined\\\hline
Source / 1 & 0/2 & 0/2 & 0/4\\
Source / 4 & 0/2 & 1/2 & 1/4\\
Adapted / 1 & 0/2 & 0/2 & 0/4\\
Adapted / 4 & 0/2 & 0/2 & 0/4\\
\end{tabular}\end{center}
All four clean references pass the fixed OCR rule. Eighteen actual draws
(sixteen supplied-content pages and two execution replays) take 794.092\,s
at 16.409\,GiB peak allocation. Both replays and reader-input comparisons
are exact; all original generated pages are authenticated and inspected.

The separately registered reference-removal control sets only the native
denoiser's \path{cond_hidden_states} argument to None. The original semantic
image/text input, prompt bytes and noise seeds remain unchanged. All 400
ablated calls omit the reference; 100 replay calls retain it. Actual hooks
verify other arguments, all eight reader-input comparisons and both first-case
noise/semantic-condition comparisons exactly. Two full-reference execution
replays are exact. Eight ablated draws plus those two replays take 336.887\,s
at 16.429\,GiB peak allocation, with no optimizer update. Source exact writing
changes from 1/4 to 0/4, and adapted writing remains 0/4. All ten new pages,
eight original counterparts and four references retain actual raw review.
Removing this reference is insufficient to repair these four cases. It also
changes appended image-patch attention geometry, so the result does not rule
out all reference interference or identify a unique repair.

\clearpage
\plate{Figure_S32_native_writing_controls}{\textbf{Reader-query and supplied-content controls retain failed writing.}
\textbf{a}, The completed query fit leaves seen/held QA and copying counts
zero. \textbf{b}, Correct-text source writing at guidance four falls from one
exact page to zero after reference removal; adapted writing stays inexact.
\textbf{c}, All four fixed source cases, in original order, include the one
successful full-reference English page. Full-page thumbnails accompany
enlarged ink crops, selected by a fixed luminance bound of 224 plus margin.
Cropped pixel values are unchanged; no contrast or glyph repair enters scoring. These are canonical OCR
counts on reused small cohorts, not a general semantic benchmark.}{fig:writing-controls}

\clearpage
\section*{Supplementary Note 56 | Direct physical-pixel likelihood and fitting}
\label{note:pixel-density}
The V68 custom image actor supplies two input-only 4,096-dimensional query
states per case. This interface is not the source's native QA interface and
must not inherit its 13/16 Chinese text-answer result. A cache of 512 training
records (256 CMRC, 256 Dolly) and 192 development records (128 CMRC in 64
article pairs, 64 Dolly) binds all states to their original prompt and target
hashes. CMRC is a Chinese question cohort; its answers may include Latin names
or digits. Chinese erasure removes only question ink; English erasure removes
all prompt ink. Normal page coordinates remain unchanged. No reader is loaded
or updated in this head probe.

A binary $32\times64$ first block is packed losslessly into 256 adjacent
$2\times4$ physical pieces. Each piece $b_j$ has 256 possible eight-bit
patterns. The exact conditional objective is
\[
\mathcal L_{\mathrm{pix}}=-\frac{1}{256}\sum_{j=1}^{256}
\log p_\theta(b_j\mid h,b_{<j}).
\]
These patterns are discrete physical outcomes, not Unicode or word identities.
Packing has no font/glyph inventory and represents all binary blocks, which
does not establish semantic knowledge of arbitrary scripts. A four-layer,
width-192, six-head causal decoder uses eight signed input bits, right-shifted
after a learned start vector, learned positions and an 8,192-to-192 contextual
projection. Training-only coordinate means and population standard deviations
(floor $10^{-4}$) normalize the state. The head has 3,453,376 parameters.
Training supplies correct preceding pieces; greedy generation uses only the
model's previous emitted patterns. Current/future target pieces cannot enter
their own logits.

The short fixed fit uses 1,200 FP32 AdamW updates, batch 32, 75 epochs and
38,400 retained draws. Training NLL falls from 5.7126 to 0.5764 nats per piece.
Autonomous exact first blocks are 0/256 CMRC training, 12/256 Dolly training,
0/128 CMRC development and 0/64 Dolly development. A subsequent CPU diagnosis
of all 896 normal/erased cases supplies correct target prefixes. Training
piece accuracy is only 76.831\% CMRC and 90.796\% Dolly; complete exact counts
remain 0/256 and 12/256. Thus generated-prefix drift alone is insufficient to
explain the shorter fit's training failure.

The longer control restores exactly the original untrained head, same data,
normalization and first 75 permutations. It uses 16,000 updates, 1,000 epochs
and 512,000 retained draws. Both use betas $(.9,.95)$, decay $.01$, gradient
clipping at one, 32-update warmup and peak/final rates $3\times10^{-4}$ and
$3\times10^{-5}$. Cosine decay spans each run's complete schedule; the
intervention therefore changes update budget and decay timing. Only final
checkpoints receive autonomous quality assessment. Matched complete-training
NLL is recorded at updates 0, 1,200, 4,000, 8,000 and 16,000.

\begin{center}\begin{tabular}{lrr}
Autonomous first blocks & Short fit & Longer fit\\\hline
CMRC training exact & 0/256 & 256/256\\
Dolly training exact & 12/256 & 256/256\\
CMRC development exact & 0/128 & 0/128\\
Dolly development exact & 0/64 & 0/64\\
CMRC development F1 & 33.498\% & 40.040\%\\
Dolly development F1 & 37.629\% & 38.867\%\\
\end{tabular}\end{center}
Longer-fit training NLL is $2.6094\times10^{-8}$ nats per piece. Chinese
question erasure reduces development F1 from 40.040\% to 31.324\%, a paired
8.716-point effect with 95\% article-bootstrap interval [6.381,11.447]
(2,000 resamples). Both questions are exact in 0/64 article pairs. English
normal/erased F1 is 38.867/25.984\%. The reused development data are not a
sealed benchmark. Neither exact fitting nor question sensitivity qualifies
semantic answering, new writing styles, historical scripts or full answers.

Longer active head training takes 279.131\,s, generation of all 896 blocks
12.721\,s and complete execution 295.602\,s, at 0.825\,GiB allocated peak.
These exclude external foundation pretraining, the earlier reader and cache,
and offline preparation/audit. They do not establish efficiency at matched
useful language quality. Source PNGs are not read by this standalone head;
complete answer histories and stopping are not evaluated.

Both completed fits pass exact packing, target-suffix causality,
cached/full-prefix generation and actual update-two/checkpoint/update-three
replay controls. The replay adds one executed optimizer call and 32 draws
per fit without retained learning exposure. Two earlier short-fit attempts
remain failures: an unsupported deterministic fused NLL kernel before any
update, and an optimizer step-counter device error during replay. Equivalent
explicit log-softmax likelihood and correct checkpoint placement repair the
implementation without changing data, criteria or tolerances. Independent
audits score every original output, verify PNG/emitted-bit identity and source
bindings. All twelve complete short/long plates are visually inspected.
These audits do not independently replay every inherited neural forward or
assign semantic scores. Full data and provenance are retained under
\path{references/evidence/pixel_density_long_fit_result/}.

\clearpage
\plate{Figure_S33_conditional_pixel_fitting}{\textbf{A small pixel decoder fits known conditions without held-out language generation.}
\textbf{a}, Cached image-derived states condition a normalized causal density
over physical pieces. \textbf{b}, The same untrained head and data reach
512/512 exact training first blocks under longer optimization, with zero exact
normal development blocks. The native chart shows matched whole-training NLL
on a logarithmic scale. \textbf{c}, Original fixed reference/short-fit/long-fit
blocks retain boundary-clipped forms. Indices follow reference content and
source order, not output quality. These are first blocks, not complete answers.}{fig:pixel-fitting}
''')
    summary=(r'Written language includes handwritten and historical forms that cannot always be specified by a character code. '
        r'We investigate an independent model that reads writing images and generates answer images, using encoded text only for offline preparation or separate diagnostics. '
        r'Controlled experiments distinguish faithful reconstruction, answer selection and autonomous writing on one consumer GPU. '
        r'A frozen continuous codec reconstructs 126 of 128 development blocks exactly, yet retaining full pretrained context gives no exact development answers after 40,000 updates. '
        r'Exact copying and constrained source-pixel answer selection do not establish composition of absent answers. '
        r'Native reader-query learning and supplied-correct-text writer controls also retain inaccurate outputs. '
        r'A separate 3.453M-parameter causal pixel head reaches 512/512 exact training first blocks after 4.65 minutes of optimization, but none of 192 development first blocks. '
        r'These head costs exclude the pretrained reader and its cached states. '
        r'Accurate emission for learned conditions is therefore achievable, while generalization from visual language to complete correct answers remains unresolved. '
        r'Historical-script understanding, general instruction following and efficiency at matched language quality remain unachieved.')
    front=(paper/'frontmatter.tex').read_text();anchor=r'\newcommand{\PaperSummary}'
    require(front.count(anchor)==1,'unique abstract command required')
    (paper/'frontmatter.tex').write_text(front[:front.index(anchor)]+anchor+'{'+summary+'}\n')


def prepare(stage):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for publication preparation')
    bound,_,_=source_evidence();before=protected()
    for name in before:require((HERE/name).read_bytes()==baseline(name),'active baseline differs: '+name)
    stage.mkdir(parents=True,exist_ok=False);paper=stage/'paper';paper.mkdir()
    for source in HERE.iterdir():
        if source.is_file() and source.suffix in ('.tex','.sty') and not source.name.startswith('redlined_'):
            shutil.copyfile(source,paper/source.name)
    shutil.copytree(HERE/'figures',paper/'figures')
    figures=append_figures(paper,stage)
    tex_fragments(paper)
    article=(paper/'article_content.tex').read_text();anchor=r'\section*{Discussion}'
    require(article.count(anchor)==1,'unique Results insertion required')
    addition=(r'\subsection*{Precise image fitting does not establish language generalization}'+'\n'
        +r'\input{writing_controls_results.tex}'+'\n\n')
    (paper/'article_content.tex').write_text(article.replace(anchor,addition+anchor))
    supplement=(paper/'supplementary_information.tex').read_text();anchor='\\clearpage\n\\section*{Reproduction}'
    require(supplement.count(anchor)==1,'unique supplementary insertion required')
    (paper/'supplementary_information.tex').write_text(supplement.replace(anchor,'\\input{writing_controls_notes.tex}\n\n'+anchor))
    require(protected()==before,'active assets changed during staged preparation')
    save(stage/'preparation.json',dict(builder_sha256=sha(__file__),baseline_commit=BASE,
        active_before_sha256=before,bound_evidence_sha256=bound,figures=figures,
        prepared_files_sha256={str(p.relative_to(paper)):sha(p) for p in sorted(paper.rglob('*')) if p.is_file()},
        active_publication_updated=False,general_ilm_goal_complete=False))
    print('Prepared48 native slides /53 charts and complete Notes54--56; no active replacement.')


def verify(stage):
    meta=load(stage/'preparation.json')
    require(meta['builder_sha256']==sha(__file__),'publication builder changed')
    require(protected()==meta['active_before_sha256'],'active publication changed')
    for name,digest in meta['bound_evidence_sha256'].items():require(sha(ROOT/name)==digest,'evidence changed: '+name)
    for name,digest in meta['prepared_files_sha256'].items():require(sha(stage/'paper'/name)==digest,'prepared asset changed: '+name)
    return meta


def render(stage):
    with build_slot():
        meta=verify(stage);folder=stage/'render';folder.mkdir(exist_ok=False)
        predecessor=stage/'predecessor.pptx';shutil.copyfile(HERE/'ILM_Figures_editable.pptx',predecessor)
        owned_command(['soffice','-env:UserInstallation='+(stage/'libreoffice-profile').as_uri(),
            '--headless','--convert-to','pdf:impress_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"}}',
            '--outdir',str(folder),str(predecessor),str(stage/'paper/ILM_Figures_editable.pptx')],
            stage,stage/'office.log',180)
        old,new=fitz.open(folder/'predecessor.pdf'),fitz.open(folder/'ILM_Figures_editable.pdf')
        require((len(old),len(new))==(46,48),'native conversion count differs')
        for i in range(46):
            a,b=old[i].get_pixmap(alpha=False),new[i].get_pixmap(alpha=False)
            require((a.width,a.height,a.samples)==(b.width,b.height,b.samples),'old slide appearance changed: '+str(i+1))
        review=fitz.open(stream=baseline('ILM_Figures_review.pdf'),filetype='pdf')
        review.insert_pdf(new,from_page=46,to_page=47);review.save(stage/'paper/ILM_Figures_review.pdf')
        figures={}
        for i,name in enumerate(FIGURES,46):
            figure=fitz.open();figure.insert_pdf(new,from_page=i,to_page=i)
            figure.save(stage/'paper/figures'/(name+'.pdf'))
            new[i].get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False).save(stage/'paper/figures'/(name+'.png'))
            (stage/'paper/figures'/(name+'.svg')).write_text(new[i].get_svg_image())
            expected={(tuple(m['RGB_size']),m['RGB_pixels_sha256']) for m in meta['figures']['media']
                if ('block_' in m['part'])==(i==47)}
            require(expected<=pdf_pixels(figure),'render changed original/cropped image pixels')
            figures[name]={suffix:sha(stage/'paper/figures'/(name+'.'+suffix)) for suffix in ('pdf','png','svg')}
        save(stage/'render.json',dict(preparation_sha256=sha(stage/'preparation.json'),prior46_appearances_exact=True,
            figure_files_sha256=figures,review_pdf_sha256=sha(stage/'paper/ILM_Figures_review.pdf'),
            original_evidence_pixels_preserved=True,PowerPoint_application_inspected=False,
            font_substitution='Arial rendered with Liberation Sans',actual_visual_review_pending=True))
        print('Rendered48 slides; all46 prior appearances match. Inspect S32/S33 before manuscript build.')


def redline(stage,source,stem):
    require(sha(TOOL)==TOOL_SHA,'pinned official latexdiff changed')
    folder=stage/'redline_inputs'/stem;folder.mkdir(parents=True,exist_ok=False)
    old,new={},{}
    (folder/'old.tex').write_text(expanded(source,baseline,old))
    (folder/'new.tex').write_text(expanded(source,lambda name:(stage/'paper'/name).read_bytes(),new))
    result=subprocess.run(['perl',str(TOOL),'--encoding=utf8','--type=CFONT','--subtype=SAFE',
        '--floattype=IDENTICAL','--math-markup=0','old.tex','new.tex'],cwd=folder,capture_output=True,text=True,check=True)
    notice=(r'\noindent{\color{blue}\footnotesize Change-marked copy relative to the preserved '
        r'46-slide writing-controls baseline (3011de6).}\par\medskip'+'\n')
    text=result.stdout.replace(r'\begin{document}',r'\begin{document}'+'\n'+notice,1)
    (stage/'paper'/(stem+'.tex')).write_text(text)
    return dict(old_inputs_sha256=old,new_inputs_sha256=new,added_text_markers=text.count(r'\DIFadd{'))


def build(stage):
    with build_slot():
        meta=verify(stage);rendered=load(stage/'render.json');review=load(stage/'figure_review.json')
        require(review['render_receipt_sha256']==sha(stage/'render.json') and review['both_new_figures_actually_viewed'],
            'actual figure review required before manuscript build')
        for name,digests in rendered['figure_files_sha256'].items():
            for suffix,digest in digests.items():require(sha(stage/'paper/figures'/(name+'.'+suffix))==digest,'rendered figure changed')
        logs=stage/'build_logs';logs.mkdir(exist_ok=False);redlines={}
        for source,stem in [('manuscript_review.tex','redlined_manuscript'),('supplementary_information.tex','redlined_supplementary_information')]:
            redlines[stem]=redline(stage,source,stem)
        outputs,pages,locations={},{},{}
        terms=('Precise image fitting','Supplementary Note 54','Supplementary Note 55','Supplementary Note 56','Fig. S32','Fig. S33','512/512','279.131')
        for stem in DOCS:
            auxiliary=stage/'aux'/stem;auxiliary.mkdir(parents=True,exist_ok=False)
            owned_command(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+str(auxiliary),stem+'.tex'],
                stage/'paper',logs/(stem+'.build.log'),180)
            log=(auxiliary/(stem+'.log')).read_text()
            require(not any(t in log for t in ('Overfull','undefined references','undefined citations','LaTeX Error')),'LaTeX/layout error: '+stem)
            path=stage/'paper'/(stem+'.pdf');shutil.copyfile(auxiliary/(stem+'.pdf'),path)
            document=fitz.open(path);text=' '.join(' '.join(p.get_text().split()) for p in document)
            require('512/512' in text and '279.131' in text,'new fitting result absent: '+stem)
            if 'supplementary' in stem:
                for term in terms[1:6]:require(term in text,'new supplement content absent: '+term)
            owned_command(['qpdf','--check',str(path)],stage/'paper',logs/(stem+'.qpdf.log'),30)
            outputs[path.name]=sha(path);pages[stem]=len(document)
            locations[stem]={term:[i+1 for i,p in enumerate(document) if term in ' '.join(p.get_text().split())] for term in terms}
            print('Built '+stem+': '+str(len(document))+' pages',flush=True)
        verify(stage)
        save(stage/'build.json',dict(preparation_sha256=sha(stage/'preparation.json'),render_receipt_sha256=sha(stage/'render.json'),
            PDFs_sha256=outputs,pages=pages,locations=locations,redlines=redlines,
            no_overfull_or_undefined=True,active_publication_updated=False,actual_pdf_review_pending=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','render','build'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();globals()[args.stage](args.output.resolve())


if __name__=='__main__':main()
