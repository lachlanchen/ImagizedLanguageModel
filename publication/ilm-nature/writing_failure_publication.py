#!/usr/bin/env python3
"""Stage, build and deliver source-bound writing/semantic-loop results."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import fitz
import openpyxl
from PIL import Image

from native_flow_manuscript import build_slot,owned_command,protected,expanded,TOOL,TOOL_SHA
from render_native_qa_publication import pdf_pixels
from writing_failure_figures import HERE,ROOT,FIGURES,append,require,sha,load,NS
from writing_failure_text import SUMMARY,RESULTS,NOTE,ARTICLE_REPLACEMENTS

BASE='947058e7'
CODE=('writing_failure_publication.py','writing_failure_figures.py','writing_failure_text.py')
DOCS=('manuscript','manuscript_review','supplementary_information',
      'redlined_manuscript','redlined_supplementary_information')
RESULT_INSERT='\\input{writing_failure_results.tex}\n\n'
NOTE_INSERT='\\input{writing_failure_note.tex}\n\n'
EVIDENCE=('pixel_language_diversity_final16000','pixel_language_first_block_likelihood',
    'pixel_language_source_reader','semantic_glyph_bridge','semantic_glyph_language',
    'semantic_glyph_counterfactual','semantic_glyph_feedback_audit','semantic_glyph_cli_cold')
NAMES=('frontmatter.tex','article_content.tex','supplementary_information.tex',
       'writing_failure_results.tex','writing_failure_note.tex',
       'redlined_manuscript.tex','redlined_supplementary_information.tex',
       *(n+'.pdf' for n in DOCS),'ILM_Figures_editable.pptx','ILM_Figures_review.pdf',
       *('figures/'+n+'.'+s for n in FIGURES for s in ('pdf','png','svg')))


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def baseline(name):
    return subprocess.check_output(['git','show',BASE+':publication/ilm-nature/'+name],cwd=ROOT)


def prepare(stage,native_from=None):
    with build_slot():
        before=protected()
        for name in before:require((HERE/name).read_bytes()==baseline(name),'active baseline changed: '+name)
        code={name:sha(HERE/name) for name in CODE}
        for name,digest in code.items():
            require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:publication/ilm-nature/'+name],cwd=ROOT)).hexdigest()==digest,
                    'commit publication builders before preparation')
        bound={}
        for name in EVIDENCE:
            folder=ROOT/'references/evidence'/name
            for p in sorted(folder.rglob('*')):
                if p.is_file():bound[str(p.relative_to(ROOT))]=sha(p)
        require(load(ROOT/'references/evidence/semantic_glyph_feedback_audit/report.json')['all_cold_GPU_embedding_hashes_bit_exact'],
                'completed feedback verification required')
        stage.mkdir(parents=True,exist_ok=False);paper=stage/'paper';paper.mkdir()
        for source in HERE.iterdir():
            if source.is_file() and source.suffix in ('.tex','.sty') and not source.name.startswith('redlined_'):
                shutil.copy2(source,paper/source.name)
        shutil.copytree(HERE/'figures',paper/'figures')
        if native_from is None:
            figures=append(paper)
        else:
            previous=load(native_from/'preparation.json')
            require(previous['code_sha256']['writing_failure_figures.py']==code['writing_failure_figures.py']
                    and previous['evidence_sha256']==bound,'figure source/evidence changed; regenerate native package')
            path=native_from/'paper/ILM_Figures_editable.pptx'
            require(sha(path)==previous['prepared_sha256']['ILM_Figures_editable.pptx'],'verified native package changed')
            shutil.copy2(path,paper/path.name);figures=previous['figures']
        (paper/'writing_failure_results.tex').write_text(RESULTS)
        (paper/'writing_failure_note.tex').write_text(NOTE)
        front=(paper/'frontmatter.tex').read_text();anchor=r'\newcommand{\PaperSummary}'
        require(front.count(anchor)==1,'unique abstract required')
        (paper/'frontmatter.tex').write_text(front[:front.index(anchor)]+anchor+'{'+SUMMARY+'}\n')
        article=(paper/'article_content.tex').read_text();anchor=r'\section*{Discussion}'
        require(article.count(anchor)==1,'unique Results/Discussion boundary required')
        article=article.replace(anchor,RESULT_INSERT+anchor)
        for old,new in ARTICLE_REPLACEMENTS:
            require(article.count(old)==1,'bounded article replacement not unique: '+old[:70])
            article=article.replace(old,new)
        (paper/'article_content.tex').write_text(article)
        supplement=(paper/'supplementary_information.tex').read_text()
        anchor='\\clearpage\n\\section*{Reproduction}'
        require(supplement.count(anchor)==1,'unique Reproduction boundary required')
        (paper/'supplementary_information.tex').write_text(supplement.replace(anchor,NOTE_INSERT+anchor))
        require(protected()==before,'active publication changed while staging')
        save(stage/'preparation.json',dict(baseline_commit=BASE,code_sha256=code,active_before_sha256=before,
            evidence_sha256=bound,figures=figures,
            prepared_sha256={str(p.relative_to(paper)):sha(p) for p in sorted(paper.rglob('*')) if p.is_file()},
            active_publication_updated=False,general_ILM_goal_complete=False))


def verify(stage):
    meta=load(stage/'preparation.json')
    require(protected()==meta['active_before_sha256'],'active publication changed')
    for name,digest in meta['code_sha256'].items():require(sha(HERE/name)==digest,'builder changed')
    for name,digest in meta['evidence_sha256'].items():require(sha(ROOT/name)==digest,'bound evidence changed')
    for name,digest in meta['prepared_sha256'].items():require(sha(stage/'paper'/name)==digest,'prepared input changed: '+name)
    return meta


def render(stage):
    with build_slot():
        meta=verify(stage);folder=stage/'render';folder.mkdir(exist_ok=False)
        predecessor=stage/'predecessor.pptx';shutil.copy2(HERE/'ILM_Figures_editable.pptx',predecessor)
        owned_command(['soffice','-env:UserInstallation='+(stage/'office-profile').as_uri(),'--headless',
            '--convert-to','pdf:impress_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"}}',
            '--outdir',str(folder),str(predecessor),str(stage/'paper/ILM_Figures_editable.pptx')],
            stage,stage/'office.log',180)
        old=fitz.open(folder/'predecessor.pdf');new=fitz.open(folder/'ILM_Figures_editable.pdf')
        require((len(old),len(new))==(50,53),'slide count changed')
        for i in range(50):
            a,b=old[i].get_pixmap(alpha=False),new[i].get_pixmap(alpha=False)
            require((a.width,a.height,a.samples)==(b.width,b.height,b.samples),'prior slide appearance changed: '+str(i+1))
        with zipfile.ZipFile(HERE/'ILM_Figures_editable.pptx') as a,zipfile.ZipFile(stage/'paper/ILM_Figures_editable.pptx') as b:
            mutable={'[Content_Types].xml','ppt/presentation.xml','ppt/_rels/presentation.xml.rels'}
            require(all(a.read(n)==b.read(n) for n in a.namelist() if n not in mutable),'prior OOXML part changed')
            for row in meta['figures']['charts']:
                node=ET.fromstring(b.read(row['part']))
                values=[float(n.text) for n in node.findall('c:chart/c:plotArea/c:barChart/c:ser/c:val/c:numRef/c:numCache/c:pt/c:v',NS)]
                book=openpyxl.load_workbook(io.BytesIO(b.read(row['workbook'])),data_only=True)
                require(values==row['values'] and all(book.active.cell(i+2,2).value==v or
                    abs(book.active.cell(i+2,2).value-v)<1e-10 for i,v in enumerate(values)),'native chart data differs')
        review=fitz.open(HERE/'ILM_Figures_review.pdf');require(len(review)==50,'review baseline count differs')
        review.insert_pdf(new,from_page=50,to_page=52);review.save(stage/'paper/ILM_Figures_review.pdf')
        expected={(tuple(Image.open(ROOT/m['source']).size),hashlib.sha256(Image.open(ROOT/m['source']).convert('RGB').tobytes()).hexdigest())
                  for m in meta['figures']['media']}
        require(expected<=pdf_pixels(new),'Office changed evidence image pixels')
        for i,name in enumerate(FIGURES,50):
            doc=fitz.open();doc.insert_pdf(new,from_page=i,to_page=i)
            doc.save(stage/'paper/figures'/(name+'.pdf'))
            new[i].get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False).save(stage/'paper/figures'/(name+'.png'))
            (stage/'paper/figures'/(name+'.svg')).write_text(new[i].get_svg_image())
        save(stage/'render.json',dict(preparation_sha256=sha(stage/'preparation.json'),
            prior50_slide_appearances_exact=True,prior54_chart_parts_preserved=True,
            three_new_chart_workbooks_exact=True,original_evidence_pixels_preserved=True,
            native_slides=53,native_charts=57,PowerPoint_application_inspected=False,
            font_substitution='Arial rendered with Liberation Sans',
            figure_files_sha256={name+'.'+s:sha(stage/'paper/figures'/(name+'.'+s)) for name in FIGURES for s in ('pdf','png','svg')},
            actual_visual_review_pending=True))


def reuse_render(stage,previous):
    with build_slot():
        verify(stage);old=load(previous/'preparation.json');receipt=load(previous/'render.json')
        require(sha(previous/'paper/ILM_Figures_editable.pptx')==old['prepared_sha256']['ILM_Figures_editable.pptx'],
                'previous rendered package changed')
        with zipfile.ZipFile(stage/'paper/ILM_Figures_editable.pptx') as a,zipfile.ZipFile(previous/'paper/ILM_Figures_editable.pptx') as b:
            require(set(a.namelist())==set(b.namelist()) and all(a.read(n)==b.read(n) for n in a.namelist()),
                    'native package content differs; rerender instead')
        for name,digest in receipt['figure_files_sha256'].items():
            require(sha(previous/'paper/figures'/name)==digest,'previous rendered figure changed')
            shutil.copy2(previous/'paper/figures'/name,stage/'paper/figures'/name)
        shutil.copy2(previous/'paper/ILM_Figures_review.pdf',stage/'paper/ILM_Figures_review.pdf')
        save(stage/'render.json',dict(receipt,preparation_sha256=sha(stage/'preparation.json'),
            reused_verified_render_from=str(previous.relative_to(ROOT)),
            reused_render_receipt_sha256=sha(previous/'render.json'),
            all_native_package_parts_bit_exact_to_rendered_source=True,actual_visual_review_pending=True))


def redline(stage,source,stem):
    require(sha(TOOL)==TOOL_SHA,'pinned latexdiff changed')
    folder=stage/'redline_inputs'/stem;folder.mkdir(parents=True,exist_ok=False)
    old,new={},{}
    (folder/'old.tex').write_text(expanded(source,baseline,old))
    (folder/'new.tex').write_text(expanded(source,lambda n:(stage/'paper'/n).read_bytes(),new))
    result=subprocess.run(['perl',str(TOOL),'--encoding=utf8','--type=CFONT','--subtype=SAFE',
        '--floattype=IDENTICAL','--math-markup=0','old.tex','new.tex'],cwd=folder,capture_output=True,text=True,check=True,timeout=90)
    notice=r'\noindent{\color{blue}\footnotesize Change-marked copy relative to the preserved 50-slide baseline (947058e7).}\par\medskip'+'\n'
    text=result.stdout.replace(r'\begin{document}',r'\begin{document}'+'\n'+notice,1)
    if source=='manuscript_review.tex':
        token=r'\newcommand{\PaperSummary}';match=re.search(r'(?m)^'+re.escape(token),text)
        require(match is not None,'redline abstract missing')
        left=text.index('{',match.start()+len(token));right=left+1;depth=1
        while depth:
            require(right<len(text),'unclosed abstract')
            if text[right] in '{}':depth+=1 if text[right]=='{' else -1
            right+=1
        text=text[:left]+'{'+r'\DIFadd{'+SUMMARY+'}}'+text[right:]
    (stage/'paper'/(stem+'.tex')).write_text(text)
    return dict(old_inputs_sha256=old,new_inputs_sha256=new,pinned_latexdiff_sha256=TOOL_SHA)


def build(stage):
    with build_slot():
        verify(stage);r=load(stage/'figure_review.json')
        require(r['render_sha256']==sha(stage/'render.json') and r['all3_new_figures_actually_viewed']
                and r.get('approved_for_build') is True,'review and approve actual new figures first')
        logs=stage/'build_logs';logs.mkdir(exist_ok=False)
        redlines={stem:redline(stage,source,stem) for source,stem in [('manuscript_review.tex','redlined_manuscript'),
            ('supplementary_information.tex','redlined_supplementary_information')]}
        pdfs={};pages={};locations={}
        for stem in DOCS:
            auxiliary=stage/'aux'/stem;auxiliary.mkdir(parents=True,exist_ok=False)
            owned_command(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+str(auxiliary),stem+'.tex'],
                stage/'paper',logs/(stem+'.log'),180)
            log=(auxiliary/(stem+'.log')).read_text()
            require(not any(t in log for t in ['Overfull','undefined references','undefined citations','LaTeX Error']),
                    'LaTeX layout/reference error: '+stem)
            pdf=stage/'paper'/(stem+'.pdf');shutil.copy2(auxiliary/(stem+'.pdf'),pdf)
            owned_command(['qpdf','--check',str(pdf)],stage/'paper',logs/(stem+'.qpdf.log'),30)
            doc=fitz.open(pdf);text=' '.join(' '.join(p.get_text().split()) for p in doc)
            require('495/512' in text and 'eleven' in text,'completed measurements absent from '+stem)
            pdfs[pdf.name]=sha(pdf);pages[stem]=len(doc)
            locations[stem]={t:[i+1 for i,p in enumerate(doc) if t in ' '.join(p.get_text().split())]
                for t in ['The learned writing interface','Supplementary Note 58','Supplementary Note 59','Fig. S36','Fig. S37','Fig. S38','0.970408','2,781']}
            print('Built '+stem+': '+str(len(doc))+' pages',flush=True)
        verify(stage)
        save(stage/'build.json',dict(PDFs_sha256=pdfs,pages=pages,locations=locations,redlines=redlines,
            preparation_sha256=sha(stage/'preparation.json'),render_sha256=sha(stage/'render.json'),
            no_overfull_or_undefined=True,actual_PDF_review_pending=True,active_publication_updated=False))


def promote(stage):
    with build_slot():
        meta=verify(stage);built=load(stage/'build.json');review=load(stage/'review.json')
        require(review['build_sha256']==sha(stage/'build.json') and review['all_changed_PDF_pages_actually_viewed'],
                'matching actual PDF review required')
        require(review['PDFs_sha256']==built['PDFs_sha256'],'reviewed PDFs differ')
        for name,digest in review['staged_files_sha256'].items():require(sha(stage/'paper'/name)==digest,'reviewed artifact changed')
        require(set(review['staged_files_sha256'])==set(NAMES),'complete promotion set required')
        article=(stage/'paper/article_content.tex').read_text().replace(RESULT_INSERT,'')
        for old,new in reversed(ARTICLE_REPLACEMENTS):
            require(article.count(new)==1,'bounded replacement changed')
            article=article.replace(new,old)
        require(article==baseline('article_content.tex').decode(),
                'article changed outside bounded insertion')
        require((stage/'paper/supplementary_information.tex').read_text().replace(NOTE_INSERT,'')==baseline('supplementary_information.tex').decode(),
                'supplement changed outside bounded insertion')
        folder=HERE/'baselines/pre-writing-failure-947058e7';folder.mkdir(exist_ok=False)
        save(folder/'baseline_manifest.json',dict(source_commit=BASE,files_sha256=meta['active_before_sha256'],recoverable_from_Git=True))
        for name in NAMES:shutil.copy2(stage/'paper'/name,HERE/name)
        for name in ('preparation','render','figure_review','build','review'):
            shutil.copy2(stage/(name+'.json'),HERE/'evidence'/('writing_failure_'+name+'.json'))
        save(stage/'promotion.json',dict(baseline_commit=BASE,files_sha256={n:sha(HERE/n) for n in NAMES},
            active_publication_updated=True,general_ILM_goal_complete=False))


def sync(stage):
    with build_slot():
        from promote_native_flow_manuscript import DEST,ALIASES,atomic_copy
        promotion=load(stage/'promotion.json')
        for n,d in promotion['files_sha256'].items():require(sha(HERE/n)==d,'promoted publication changed')
        require(DEST.is_dir(),'authorized Nutstore ILM folder absent')
        package=DEST/'ILM_NatureStyle_20261110_SemanticGlyphLoop';package.mkdir(exist_ok=False)
        files={}
        for p in sorted((stage/'paper').rglob('*')):
            if p.is_file():
                relative=p.relative_to(stage/'paper');atomic_copy(p,package/relative);files[str(relative)]=sha(package/relative)
        aliases={}
        for name,alias in ALIASES.items():
            existing=DEST/alias
            if existing.exists():
                require(sha(existing) in [load(stage/'preparation.json')['active_before_sha256'][name],sha(HERE/name)],'Nutstore alias has a different revision')
            atomic_copy(HERE/name,existing);require(sha(existing)==sha(HERE/name),'local sync differs');aliases[alias]=sha(existing)
        receipt=dict(package=package.name,files_sha256=files,aliases_sha256=aliases,
            local_hashes_verified=True,cloud_propagation_verified=False)
        save(stage/'sync.json',receipt);save(HERE/'evidence/writing_failure_sync.json',receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','render','reuse-render','build','promote','sync'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--render-from',type=Path)
    parser.add_argument('--native-from',type=Path);args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output.resolve(),args.native_from.resolve() if args.native_from else None)
    elif args.stage=='reuse-render':
        require(args.render_from is not None,'supply prior verified render')
        reuse_render(args.output.resolve(),args.render_from.resolve())
    else:globals()[args.stage](args.output.resolve())
