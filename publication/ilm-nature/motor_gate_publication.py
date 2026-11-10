#!/usr/bin/env python3
"""Stage/build/promote completed motor controls while preserving the active paper."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile

import fitz
import openpyxl
from PIL import Image

from native_flow_manuscript import build_slot,owned_command,protected,expanded,TOOL,TOOL_SHA
from render_native_qa_publication import pdf_pixels
from motor_gate_data import HERE,ROOT,load,sha,require
from motor_gate_figures import append,FIGURE,SOURCES,source_data,NS
from motor_gate_text import RESULTS,NOTE

BASE='2ca07ea0'
DOCS=('manuscript','manuscript_review','supplementary_information','redlined_manuscript','redlined_supplementary_information')
CODE=('motor_gate_publication.py','motor_gate_data.py','motor_gate_figures.py','motor_gate_text.py',
    'native_flow_manuscript.py','generate_assignment_editable_figures.py','generate_assignment_evidence.py',
    'prepare_native_qa_publication.py','render_native_qa_publication.py','promote_native_flow_manuscript.py')
NAMES=('article_content.tex','supplementary_information.tex','motor_gate_results.tex','motor_gate_note.tex',
    'redlined_manuscript.tex','redlined_supplementary_information.tex',*(n+'.pdf' for n in DOCS),
    'ILM_Figures_editable.pptx','ILM_Figures_review.pdf',*('figures/'+FIGURE+'.'+s for s in ('pdf','png','svg')))
RESULT_INSERT='\\input{motor_gate_results.tex}\n\n'
NOTE_INSERT='\\input{motor_gate_note.tex}\n\n'
EXTRA=('semantic_readout_distillation_v1/calibrate/report.json',
    'semantic_readout_distillation_v1/visual_replay/report.json',
    'semantic_readout_distillation_v1/visual_review.json',
    'contrastive_glyph_readout_v1/visual_replay/report.json',
    'contrastive_glyph_readout_v1/retention.json',
    'glyph_motor_states_v1/report.json','glyph_motor_states_v1/visual_review.json',
    'cached_glyph_teacher_audit_v1/report.json','cached_glyph_teacher_audit_v1/records.json',
    'cached_glyph_teacher_audit_v1/cases.json')


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def baseline(name):return subprocess.check_output(['git','show',BASE+':publication/ilm-nature/'+name],cwd=ROOT)


def measure_check(data):
    for value in [r[k] for r in data['motor_tests'] for k in ['clean_f1','mixed_f1']]:
        require(f'{value:.6f}' in NOTE,'measured motor value absent from note')
    require(f"{data['regression']['trained_span_mse']:.6f}" in NOTE and
            f"{data['regression']['best_span_mse']:.6f}" in NOTE,'measured rank bound absent')
    for version in data['compilers'].values():
        for row in version.values():require(f"{row['mean_f1']:.6f}" in NOTE,'compiler value absent from note')
    teacher=load(ROOT/'references/evidence/cached_glyph_teacher_audit_v1/report.json')
    plain=NOTE.replace(',','')
    for row in teacher['groups'].values():
        require(f"{row['reference_form_matches']}/{row['states']}" in plain,
                'cached-label denominator absent from note')
    for key in ['1','2-3']:
        row=teacher['prefix_bins'][key]
        require(f"{row['reference_form_matches']}/{row['states']}" in plain,
                'prefix comparison absent from note')
    require(f"{teacher['cases_all_reference_forms_and_stop_match']}/{teacher['training_questions']}" in plain
            and teacher['model_forward_calls']==0 and not teacher['semantic_accuracy_measured'],
            'cached-label audit scope differs')
    # These results remain the only completed language evidence. New glyph
    # component controls must not silently change the abstract.
    require('genuine' not in RESULTS.lower(),'unfinished genuine-curriculum result entered main text')


def prepare(stage):
    with build_slot():
        data=source_data();measure_check(data);before=protected()
        for n in before:require((HERE/n).read_bytes()==baseline(n),'active reviewed baseline changed: '+n)
        code={n:sha(HERE/n) for n in CODE}
        for n,d in code.items():
            require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:publication/ilm-nature/'+n],cwd=ROOT)).hexdigest()==d,'commit builder first')
        bound=dict(data['source_files_sha256'])
        for p in sorted(SOURCES.iterdir()):
            if p.is_file():bound[str(p.relative_to(ROOT))]=sha(p)
        for n in EXTRA:
            p=ROOT/'references/evidence'/n;bound[str(p.relative_to(ROOT))]=sha(p)
        for n in ['scripts/audit_cached_glyph_teacher_labels.py',
                  'references/cached_glyph_teacher_agreement_result.md']:
            bound[n]=sha(ROOT/n)
        for n in ['semantic_readout_distillation_protocol.md','contrastive_glyph_readout_protocol.md',
                  'robust_glyph_decoder_protocol.md','semantic_motor_flow_protocol.md','spatial_semantic_motor_protocol.md']:
            p=ROOT/'references'/n;bound[str(p.relative_to(ROOT))]=sha(p)
        stage.mkdir(parents=True,exist_ok=False);paper=stage/'paper';paper.mkdir()
        for p in HERE.iterdir():
            if p.is_file() and p.suffix in ('.tex','.sty') and not p.name.startswith('redlined_'):shutil.copy2(p,paper/p.name)
        shutil.copytree(HERE/'figures',paper/'figures')
        figures=append(paper)
        (paper/'motor_gate_results.tex').write_text(RESULTS);(paper/'motor_gate_note.tex').write_text(NOTE)
        article=(paper/'article_content.tex').read_text();anchor=r'\section*{Discussion}'
        require(article.count(anchor)==1,'unique Discussion boundary required')
        (paper/'article_content.tex').write_text(article.replace(anchor,RESULT_INSERT+anchor))
        supplement=(paper/'supplementary_information.tex').read_text();anchor='\\clearpage\n\\section*{Reproduction}'
        require(supplement.count(anchor)==1,'unique Reproduction boundary required')
        (paper/'supplementary_information.tex').write_text(supplement.replace(anchor,NOTE_INSERT+anchor))
        require(protected()==before,'active publication changed during staging')
        save(stage/'preparation.json',dict(baseline_commit=BASE,code_sha256=code,active_before_sha256=before,
            evidence_sha256=bound,figures=figures,abstract_unchanged=True,genuine_curriculum_results_included=False,
            prepared_sha256={str(p.relative_to(paper)):sha(p) for p in sorted(paper.rglob('*')) if p.is_file()},
            delivery_date=time.strftime('%Y%m%d'),active_publication_updated=False,general_ILM_goal_complete=False))


def verify(stage):
    m=load(stage/'preparation.json')
    require(protected()==m['active_before_sha256'],'active reviewed publication changed')
    for n,d in m['code_sha256'].items():require(sha(HERE/n)==d,'builder changed: '+n)
    for n,d in m['evidence_sha256'].items():require(sha(ROOT/n)==d,'bound evidence changed: '+n)
    for n,d in m['prepared_sha256'].items():require(sha(stage/'paper'/n)==d,'prepared artifact changed: '+n)
    return m


def render(stage):
    with build_slot():
        meta=verify(stage);folder=stage/'render';folder.mkdir(exist_ok=False)
        predecessor=stage/'predecessor.pptx';shutil.copy2(HERE/'ILM_Figures_editable.pptx',predecessor)
        owned_command(['soffice','-env:UserInstallation='+(stage/'office-profile').as_uri(),'--headless','--convert-to',
            'pdf:impress_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"}}',
            '--outdir',str(folder),str(predecessor),str(stage/'paper/ILM_Figures_editable.pptx')],stage,stage/'office.log',180)
        old=fitz.open(folder/'predecessor.pdf');new=fitz.open(folder/'ILM_Figures_editable.pdf')
        require((len(old),len(new))==(53,54),'actual slide counts differ')
        for i in range(53):
            a,b=old[i].get_pixmap(alpha=False),new[i].get_pixmap(alpha=False)
            require((a.width,a.height,a.samples)==(b.width,b.height,b.samples),'old slide appearance changed')
        with zipfile.ZipFile(HERE/'ILM_Figures_editable.pptx') as a,zipfile.ZipFile(stage/'paper/ILM_Figures_editable.pptx') as b:
            mutable={'[Content_Types].xml','ppt/presentation.xml','ppt/_rels/presentation.xml.rels'}
            require(all(a.read(n)==b.read(n) for n in a.namelist() if n not in mutable),'prior OOXML part changed')
            for row in meta['figures']['charts']:
                node=ET.fromstring(b.read(row['part']))
                values=[float(n.text) for n in node.findall('c:chart/c:plotArea/c:barChart/c:ser/c:val/c:numRef/c:numCache/c:pt/c:v',NS)]
                book=openpyxl.load_workbook(io.BytesIO(b.read(row['workbook'])),data_only=True)
                require(values==row['values'] and all(abs(book.active.cell(i+2,2).value-v)<1e-10 for i,v in enumerate(values)),
                        'native chart/workbook data changed')
        expected={(tuple(Image.open(ROOT/r['source']).size),hashlib.sha256(Image.open(ROOT/r['source']).convert('RGB').tobytes()).hexdigest())
            for r in meta['figures']['media']}
        require(expected<=pdf_pixels(new),'rendered PDF changed raw glyph pixels')
        review=fitz.open(HERE/'ILM_Figures_review.pdf');require(len(review)==53,'review baseline differs')
        review.insert_pdf(new,from_page=53,to_page=53);review.save(stage/'paper/ILM_Figures_review.pdf')
        doc=fitz.open();doc.insert_pdf(new,from_page=53,to_page=53);doc.save(stage/'paper/figures'/(FIGURE+'.pdf'))
        new[53].get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False).save(stage/'paper/figures'/(FIGURE+'.png'))
        (stage/'paper/figures'/(FIGURE+'.svg')).write_text(new[53].get_svg_image())
        save(stage/'render.json',dict(preparation_sha256=sha(stage/'preparation.json'),native_slides=54,native_charts=59,
            prior53_slide_appearances_exact=True,prior57_chart_parts_exact=True,two_new_native_chart_workbooks_exact=True,
            all_original_glyph_pixels_preserved=True,PowerPoint_application_inspected=False,
            font_substitution='Arial rendered with Liberation Sans',actual_visual_review_pending=True,
            figure_files_sha256={FIGURE+'.'+s:sha(stage/'paper/figures'/(FIGURE+'.'+s)) for s in ('pdf','png','svg')}))


def redline(stage,source,stem):
    require(sha(TOOL)==TOOL_SHA,'pinned latexdiff changed')
    folder=stage/'redline_inputs'/stem;folder.mkdir(parents=True,exist_ok=False);old,new={},{}
    (folder/'old.tex').write_text(expanded(source,baseline,old))
    (folder/'new.tex').write_text(expanded(source,lambda n:(stage/'paper'/n).read_bytes(),new))
    result=subprocess.run(['perl',str(TOOL),'--encoding=utf8','--type=CFONT','--subtype=SAFE','--floattype=IDENTICAL',
        '--math-markup=0','old.tex','new.tex'],cwd=folder,capture_output=True,text=True,check=True,timeout=90)
    notice=r'\noindent{\color{blue}\footnotesize Change-marked copy relative to the preserved 53-slide baseline (2ca07ea0).}\par\medskip'+'\n'
    (stage/'paper'/(stem+'.tex')).write_text(result.stdout.replace(r'\begin{document}',r'\begin{document}'+'\n'+notice,1))
    return dict(old_inputs_sha256=old,new_inputs_sha256=new,pinned_latexdiff_sha256=TOOL_SHA)


def build(stage):
    with build_slot():
        verify(stage);review=load(stage/'figure_review.json')
        require(review['render_sha256']==sha(stage/'render.json') and review['new_figure_actually_viewed']
            and review['approved_for_build'],'review actual S39 PNG before building')
        logs=stage/'build_logs';logs.mkdir(exist_ok=False)
        redlines={stem:redline(stage,source,stem) for source,stem in [('manuscript_review.tex','redlined_manuscript'),
            ('supplementary_information.tex','redlined_supplementary_information')]}
        pdfs={};pages={};locations={}
        for stem in DOCS:
            aux=stage/'aux'/stem;aux.mkdir(parents=True,exist_ok=False)
            owned_command(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+str(aux),stem+'.tex'],
                stage/'paper',logs/(stem+'.log'),180)
            log=(aux/(stem+'.log')).read_text()
            require(not any(t in log for t in ['Overfull','undefined references','undefined citations','LaTeX Error']),
                'new layout/reference error: '+stem)
            pdf=stage/'paper'/(stem+'.pdf');shutil.copy2(aux/(stem+'.pdf'),pdf)
            owned_command(['qpdf','--check',str(pdf)],stage/'paper',logs/(stem+'.qpdf.log'),30)
            doc=fitz.open(pdf);text=' '.join(' '.join(p.get_text().split()) for p in doc)
            require('0.305180' in text and '0.810537' in text and '4,087' in text,'new measured controls absent')
            pdfs[pdf.name]=sha(pdf);pages[stem]=len(doc)
            locations[stem]={token:[i+1 for i,p in enumerate(doc) if token in ' '.join(p.get_text().split())]
                for token in ['Sharp known-form generation','Supplementary Note 60','Fig. S39','0.305180','0.810537']}
            print('Built '+stem+': '+str(len(doc))+' pages',flush=True)
        verify(stage)
        save(stage/'build.json',dict(PDFs_sha256=pdfs,pages=pages,locations=locations,redlines=redlines,
            preparation_sha256=sha(stage/'preparation.json'),render_sha256=sha(stage/'render.json'),
            no_overfull_or_undefined=True,actual_PDF_review_pending=True,active_publication_updated=False))


def promote(stage):
    with build_slot():
        meta=verify(stage);built=load(stage/'build.json');review=load(stage/'review.json')
        require(review['build_sha256']==sha(stage/'build.json') and review['all_changed_PDF_pages_actually_viewed'],
                'matching actual PDF inspection required')
        require(review['PDFs_sha256']==built['PDFs_sha256'] and set(review['staged_files_sha256'])==set(NAMES),'incomplete reviewed artifacts')
        for n,d in review['staged_files_sha256'].items():require(sha(stage/'paper'/n)==d,'reviewed artifact changed')
        require((stage/'paper/article_content.tex').read_text().replace(RESULT_INSERT,'')==baseline('article_content.tex').decode(),
            'unplanned main-text edit')
        require((stage/'paper/supplementary_information.tex').read_text().replace(NOTE_INSERT,'')==baseline('supplementary_information.tex').decode(),
            'unplanned supplement edit')
        require((stage/'paper/frontmatter.tex').read_bytes()==baseline('frontmatter.tex'),'abstract changed')
        folder=HERE/'baselines/pre-motor-gate-2ca07ea0';folder.mkdir(exist_ok=False)
        save(folder/'baseline_manifest.json',dict(source_commit=BASE,files_sha256=meta['active_before_sha256'],recoverable_from_Git=True))
        for n in NAMES:shutil.copy2(stage/'paper'/n,HERE/n)
        for n in ['preparation','render','figure_review','build','review']:shutil.copy2(stage/(n+'.json'),HERE/'evidence'/('motor_gate_'+n+'.json'))
        save(stage/'promotion.json',dict(files_sha256={n:sha(HERE/n) for n in NAMES},baseline_commit=BASE,
            active_publication_updated=True,general_ILM_goal_complete=False))


def sync(stage):
    with build_slot():
        from promote_native_flow_manuscript import DEST,ALIASES,atomic_copy
        promotion=load(stage/'promotion.json');preparation=load(stage/'preparation.json')
        for n,d in promotion['files_sha256'].items():require(sha(HERE/n)==d,'promoted file changed')
        require(DEST.is_dir(),'authorized Nutstore ILM folder absent')
        package=DEST/('ILM_NatureStyle_'+preparation['delivery_date']+'_MotorAmbiguity');package.mkdir(exist_ok=False)
        files={}
        for p in sorted((stage/'paper').rglob('*')):
            if p.is_file():
                relative=p.relative_to(stage/'paper');atomic_copy(p,package/relative);files[str(relative)]=sha(package/relative)
        aliases={}
        for n,alias in ALIASES.items():
            p=DEST/alias
            if p.exists():require(sha(p) in [preparation['active_before_sha256'][n],sha(HERE/n)],'Nutstore alias has another revision')
            atomic_copy(HERE/n,p);require(sha(p)==sha(HERE/n),'local alias differs');aliases[alias]=sha(p)
        receipt=dict(package=package.name,files_sha256=files,aliases_sha256=aliases,local_hashes_verified=True,cloud_propagation_verified=False)
        save(stage/'sync.json',receipt);save(HERE/'evidence/motor_gate_sync.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','render','build','promote','sync'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();globals()[a.stage](a.output.resolve())
