#!/usr/bin/env python3
"""Prepare/build a source-bound native-flow manuscript without active writes."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import fitz

from prepare_native_qa_publication import BASE, HERE, ROOT, baseline, sha
from render_native_flow_publication import verify as verify_figures

QA = ROOT/'artifacts/native_qa_publication_stage_v1'
FLOW = ROOT/'artifacts/native_flow_publication_stage_v5'
FIGURES = ('Figure_S29_native_qa_learning','Figure_S30_native_qa_pages','Figure_S31_native_visual_flow')
DOCS = ('manuscript','manuscript_review','supplementary_information',
        'redlined_manuscript','redlined_supplementary_information')
TOOL = ROOT/'artifacts/external/latexdiff-1.4.0/latexdiff'
TOOL_SHA = 'bbf7567f48ae6d52887ebce0a390eac0718508b5e1e3ee158bffd4280dca8723'
INPUT = re.compile(r'\\input\{([^{}]+)\}')


def require(value, message):
    if not value:raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def protected():
    files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.tex','.sty','.pdf','.pptx')]
    files.extend(p for p in (HERE/'figures').rglob('*') if p.is_file())
    return {str(p.relative_to(HERE)):sha(p) for p in sorted(files)}


def prepare(stage):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for paper preparation')
    verify_figures(FLOW,QA)
    reviewed=load(ROOT/'references/evidence/native_flow_complete_publication_review.json')
    require(reviewed['visual_review_completed'] and reviewed['preparation_sha256']==sha(FLOW/'preparation.json')
        and reviewed['render_receipt_sha256']==sha(FLOW/'render/render.json'), 'reviewed figure stage required')
    before=protected()
    inputs=[p for p in HERE.iterdir() if p.suffix in ('.tex','.sty') and not p.name.startswith('redlined_')]
    for source in inputs:
        require(source.read_bytes()==baseline(source.name),'active manuscript input differs from declared baseline: '+source.name)
    stage.mkdir(parents=True,exist_ok=False);paper=stage/'paper';paper.mkdir()
    for source in inputs:shutil.copyfile(source,paper/source.name)
    shutil.copytree(HERE/'figures',paper/'figures')
    shutil.copyfile(FLOW/'draft.pptx',paper/'ILM_Figures_editable.pptx')
    shutil.copyfile(FLOW/'render/ILM_Figures_review.pdf',paper/'ILM_Figures_review.pdf')
    for name in FIGURES:
        for suffix in ('pdf','png','svg'):
            shutil.copyfile(FLOW/'render'/(name+'.'+suffix),paper/'figures'/(name+'.'+suffix))
    shutil.copyfile(QA/'results_paragraph.tex',paper/'native_qa_results.tex')
    shutil.copyfile(FLOW/'results_native_flow.tex',paper/'native_flow_results.tex')
    article=(paper/'article_content.tex').read_text()
    anchor=r'\section*{Discussion}'
    require(article.count(anchor)==1,'unique Discussion insertion point required')
    addition=(r'\subsection*{Full-page answer learning remains unsuccessful}'+'\n'
        +r'\input{native_qa_results.tex}'+'\n'+r'\input{native_flow_results.tex}'+'\n\n')
    (paper/'article_content.tex').write_text(article.replace(anchor,addition+anchor))
    qa_note='\\clearpage\n'+(QA/'supplementary_note_52.tex').read_text()+r'''

\clearpage
\plate{Figure_S29_native_qa_learning}{\textbf{Complete natural-QA adaptation fails the prospective conditions.}
\textbf{a}, Matched source-accepted OCR-exact answers after 128 updates on
32 questions remain zero on seen questions, held-out questions and complete
held-out pairs. Erased controls also give zero. \textbf{b}, Exact copying
increases by one seen string but loses three held-out strings. The count
labels show the complete cohorts. The inherited pretrained image writer,
all original failures and imperfect reference OCR are retained.}{fig:native-qa-learning}

\clearpage
\plate{Figure_S30_native_qa_pages}{\textbf{Fixed held-out inputs and unrepaired answer pages.}
\textbf{a}, Complete first held-out question pair and cropped source-question
insets for display only. \textbf{b}, Complete initial and final answer pages
for normal, question-erased and passage-erased inputs. Reference whitespace
alone is cropped for legibility. All twelve model pages remain complete;
no transcription, contrast repair or replacement writing is applied.
This fixed pair is selected by original order, not by output appearance.}{fig:native-qa-pages}
'''
    flow_note=(FLOW/'supplementary_note_53.tex').read_text()
    flow_note=re.sub(r'\\texttt\{(references/[^{}]+)\}',
        lambda m:r'\path{'+m[1].replace(r'\_','_')+'}',flow_note)
    flow_note='\\clearpage\n'+flow_note+r'''

\clearpage
\plate{Figure_S31_native_visual_flow}{\textbf{Native image-answer flow executes within one GPU but fails answering.}
\textbf{a}, A complete original input supplies frozen reader features and
a separate frozen-VAE reference latent. Connector and writer adapters learn;
the frozen VAE decodes the sampled answer. The first fixed held-out output
is shown unrepaired. \textbf{b}, Answer-image supervision defines the native
velocity loss; inference starts from independent noise and uses fifty steps.
\textbf{c}, Matched initial/final QA counts remain zero, while copying rises
from zero to five of 64 strings. All 384 original assessment pages are
authenticated and inspected. The 37.381M trainable adapters remain inside a
6.240B-parameter pretrained runtime; these costs do not establish efficient
language learning at matched quality. Reference OCR remains imperfect.}{fig:native-visual-flow}
'''
    (paper/'native_qa_note.tex').write_text(qa_note)
    (paper/'native_flow_note.tex').write_text(flow_note)
    supplement=(paper/'supplementary_information.tex').read_text()
    anchor='\\clearpage\n\\section*{Reproduction}'
    require(supplement.count(anchor)==1,'unique Reproduction insertion point required')
    addition='\\input{native_qa_note.tex}\n\\input{native_flow_note.tex}\n\n'
    (paper/'supplementary_information.tex').write_text(supplement.replace(anchor,addition+anchor))
    require(protected()==before,'active files changed during preparation')
    files={str(p.relative_to(paper)):sha(p) for p in sorted(paper.rglob('*')) if p.is_file()}
    save(stage/'preparation.json',dict(builder_sha256=sha(__file__),baseline_commit=BASE,
        active_publication_sha256=before,prepared_files_sha256=files,
        source_QA_preparation_sha256=sha(QA/'preparation.json'),
        source_flow_preparation_sha256=sha(FLOW/'preparation.json'),
        source_flow_render_sha256=sha(FLOW/'render/render.json'),
        source_flow_visual_review_sha256=sha(ROOT/'references/evidence/native_flow_complete_publication_review.json'),
        title_abstract_introduction_unchanged=True,all_prior_figure_bytes_unchanged=True,
        active_publication_updated=False,query_fit_result_included=False,built=False,general_ilm_goal_complete=False))
    print('Prepared complete isolated article/review/supplement with Notes52/53 and reviewed S29--S31.')


def verify(stage):
    meta=load(stage/'preparation.json')
    require(meta['builder_sha256']==sha(__file__),'paper builder changed after preparation')
    require(protected()==meta['active_publication_sha256'],'active publication changed')
    require(meta['source_flow_preparation_sha256']==sha(FLOW/'preparation.json')
        and meta['source_flow_render_sha256']==sha(FLOW/'render/render.json'), 'figure source changed')
    for name,digest in meta['prepared_files_sha256'].items():
        require(sha(stage/'paper'/name)==digest,'prepared source or figure changed: '+name)
    return meta


@contextmanager
def build_slot():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for CPU publication builds')
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        memory={s.split(':')[0]:int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
        swap=(memory['SwapTotal']-memory['SwapFree'])/memory['SwapTotal'] if memory['SwapTotal'] else 0.
        require(memory['MemAvailable']>=24*1024**2 and swap<=.75,'existing RAM/swap thresholds reject build')
        yield dict(available_RAM_GiB=memory['MemAvailable']/1024**2,swap_used_fraction=swap,
            same_project_GPU_lock_held=True,CUDA_hidden=True,Office_started=False)


def owned_command(command,cwd,log,timeout=180):
    with Path(log).open('x') as handle:
        process=subprocess.Popen(command,cwd=cwd,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            process.wait(timeout=timeout)
            require(process.returncode==0,'owned command failed: '+str(log))
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def expanded(name,reader,seen,ancestors=()):
    require(name not in ancestors and not Path(name).is_absolute() and '..' not in Path(name).parts,
        'external/cyclic TeX input')
    raw=reader(name);seen[name]=hashlib.sha256(raw).hexdigest()
    def child(match):
        path=str(Path(name).parent/match[1])
        if not Path(path).suffix:path+='.tex'
        return expanded(path,reader,seen,(*ancestors,name))
    return INPUT.sub(child,raw.decode()).replace(r'\balance',r'\relax')


def redline(stage,document,stem):
    require(sha(TOOL)==TOOL_SHA,'pinned official latexdiff required')
    folder=stage/'redline_inputs'/stem;folder.mkdir(parents=True,exist_ok=False)
    old_hashes,new_hashes={},{}
    old=expanded(document,baseline,old_hashes)
    new=expanded(document,lambda n:(stage/'paper'/n).read_bytes(),new_hashes)
    (folder/'old.tex').write_text(old);(folder/'new.tex').write_text(new)
    result=subprocess.run(['perl',str(TOOL),'--encoding=utf8','--type=CFONT','--subtype=SAFE',
        '--floattype=IDENTICAL','--math-markup=0','old.tex','new.tex'],cwd=folder,
        capture_output=True,text=True,check=True,timeout=90)
    (folder/'latexdiff.log').write_text(result.stderr)
    notice=(r'\noindent{\color{blue}\footnotesize Change-marked copy: additions relative to the '
        r'preserved 43-slide manuscript baseline (daef4bf) are blue.}\par\medskip'+'\n')
    source=result.stdout.replace(r'\begin{document}',r'\begin{document}'+'\n'+notice,1)
    require(source.count(r'\DIFadd{')>0,'redline has no visible added text')
    (stage/'paper'/(stem+'.tex')).write_text(source)
    return dict(old_inputs_sha256=old_hashes,new_inputs_sha256=new_hashes,
        added_text_markers=source.count(r'\DIFadd{'),latexdiff_sha256=TOOL_SHA,
        markup='CFONT / SAFE / IDENTICAL floats; math-markup=0; final columns unbalanced')


def build(stage):
    with build_slot() as resources:
        meta=verify(stage);paper=stage/'paper';logs=stage/'build_logs';logs.mkdir(exist_ok=False)
        marked={}
        for source,stem in [('manuscript_review.tex','redlined_manuscript'),
                ('supplementary_information.tex','redlined_supplementary_information')]:
            marked[stem]=redline(stage,source,stem)
        outputs={};pages={};locations={}
        for stem in DOCS:
            folder=stage/'aux'/stem;folder.mkdir(parents=True,exist_ok=False)
            owned_command(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',
                '-outdir='+str(folder),stem+'.tex'],paper,logs/(stem+'.build.log'))
            log=(folder/(stem+'.log')).read_text()
            require(not any(v in log for v in ('Overfull','undefined references','undefined citations','LaTeX Error')),
                'layout or reference problem: '+stem)
            target=paper/(stem+'.pdf');shutil.copyfile(folder/(stem+'.pdf'),target)
            document=fitz.open(target);text=' '.join(' '.join(page.get_text().split()) for page in document)
            for term in ('12,782.358','15.697','1,536','0/32','0/16'):
                require(term in text,'compiled paper lacks completed result '+term+': '+stem)
            if 'supplementary' in stem:
                for term in ('Supplementary Note 52','Supplementary Note 53','Figure S29','Figure S30','Figure S31'):
                    require(term in text,'compiled supplement lacks '+term)
            if stem.startswith('redlined_'):
                require('43-slide manuscript baseline' in text,'redline baseline notice absent')
            owned_command(['qpdf','--check',str(target)],paper,logs/(stem+'.qpdf.log'),timeout=30)
            outputs[stem+'.pdf']=sha(target);pages[stem]=len(document)
            locations[stem]={term:[i+1 for i,page in enumerate(document) if term in ' '.join(page.get_text().split())]
                for term in ('Full-page answer learning remains unsuccessful','Supplementary Note 52',
                    'Supplementary Note 53','12,782.358','15.697','Figure S29','Figure S30','Figure S31')}
            print('Built '+stem+': '+str(len(document))+' pages',flush=True)
        require(protected()==meta['active_publication_sha256'],'active files changed during isolated build')
        verify(stage)
        save(stage/'build.json',dict(builder_sha256=sha(__file__),preparation_sha256=sha(stage/'preparation.json'),
            PDFs_sha256=outputs,pages=pages,locations=locations,redlines=marked,resources=resources,
            all_compiled_without_overfull_or_undefined=True,all_PDF_structure_checks_passed=True,
            active_publication_unchanged=True,active_publication_updated=False,
            raw_PDF_page_review_completed=False,Nutstore_updated=False,general_ilm_goal_complete=False))
        print('Five isolated PDFs built. Render/review added and reflowed pages before any promotion.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','build'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();{'prepare':prepare,'build':build}[args.stage](args.output.resolve())
