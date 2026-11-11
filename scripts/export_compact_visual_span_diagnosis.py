#!/usr/bin/env python3
"""Publish bounded diagnostic evidence and measured scientific plots."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ilm.image_edit.baseline import ROOT, save_json, sha

SOURCE=ROOT/'artifacts/compact_visual_span_diagnosis_analysis_v3'
RUN=ROOT/'artifacts/compact_visual_span_diagnosis_v2'
DEST=ROOT/'references/evidence/compact_visual_span_diagnosis_20261111'
RECEIPT=DEST.with_suffix('.json')
ARMS=['codec','visual_reference_8','visual_reference_64','native_reference_8','shuffled_reference_8','emitted_fixed_counts_8','learned_prompt_emitted_8']
LABELS=['Reference\ncodec','Correct image\nhistory · 8','Correct image\nhistory · 64','Native\nhistory · 8','Wrong\ncontext · 8','Generated\nhistory · 8','Learned grouping\n+ generated · 8']


def load(path):return json.loads(Path(path).read_text())


def aggregate(rows):
    keys=['glyphs','exact_glyphs','tp','fp','fn','nonblank_target_glyphs','nonblank_exact_glyphs','blank_target_glyphs','blank_exact_glyphs']
    d={k:sum(r[k] for r in rows) for k in keys}
    d['foreground_f1']=2*d['tp']/max(2*d['tp']+d['fp']+d['fn'],1)
    return d


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU export only')
    x=load(SOURCE/'summary.json');request=load(RUN/'registration.json')
    if x['registration_sha256']!=sha(RUN/'registration.json'):raise ValueError('Registration differs')
    if not x['original_pixel_metrics_independently_recomputed'] or len(x['records'])!=64:raise ValueError('Complete authenticated pixel analysis required')
    DEST.mkdir(parents=True,exist_ok=True)
    shutil.copytree(SOURCE/'raw_examples',DEST/'raw_examples',dirs_exist_ok=True)
    for row in x['raw_examples']:
        if sha(DEST/row['file'])!=row['sha256']:raise ValueError('Copied original PNG differs')
    for name in ['pixel_metrics.csv','reader_metrics.csv']:shutil.copyfile(SOURCE/name,DEST/name)
    # Process inventories and absolute runtime paths remain in private receipts.
    public=deepcopy(x)
    public['resources']['admission'].pop('project_processes',None)
    for model in public['resources']['models']:
        codec=model['provenance']['codec']
        codec['path']=str(Path(codec['path']).relative_to(ROOT))
    table=[];first=[]
    for model in ['v1','wider']:
        for split in ['train','development']:
            for arm in ARMS:
                rows=[r for r in x['metrics'] if r['model']==model and r['split']==split and r['arm']==arm and r['part']=='all']
                table.append(dict(model=model,split=split,arm=arm,**aggregate(rows)))
            for arm in ['visual_reference_8','emitted_fixed_counts_8']:
                rows=[r for r in x['metrics'] if r['model']==model and r['split']==split and r['arm']==arm and r['part']=='first']
                first.append(dict(model=model,split=split,arm=arm,**aggregate(rows)))
    old=load(ROOT/'artifacts/compact_visual_span_diagnosis_v1/execution/status.json')
    probe=load(ROOT/'artifacts/compact_prefix_consistency_v1/execution/report.json')
    preps=[load(ROOT/f'artifacts/compact_visual_span_diagnosis_v{i}/registration.json')['preparation_seconds'] for i in [1,2]]
    public.update(aggregate_pixel_metrics=table,aggregate_first_span_metrics=first,
                  original_analysis_sha256=sha(SOURCE/'summary.json'),original_neural_report_sha256=sha(RUN/'execution/report.json'),
                  export_source_sha256=sha(Path(__file__)),request=request,
                  accounted_stage_costs=dict(stopped_neural_seconds=old['elapsed_seconds'],precision_probe_seconds=probe['elapsed_seconds'],
                      completed_neural_seconds=x['resources']['elapsed_seconds'],cpu_preparation_seconds=preps,
                      sum_registered_stage_seconds=sum(preps)+old['elapsed_seconds']+probe['elapsed_seconds']+x['resources']['elapsed_seconds'],
                      stopped_attempt_peak_vram_unavailable=True,complete_neural_peak_vram_gib=x['resources']['peak_allocated_vram_gib'],
                      precision_probe_peak_vram_gib=probe['peak_allocated_vram_gib']),
                  result='Conditional writing fails before emitted-history drift; near-exact codec control and finer sampling do not repair it. Reader native-target alignment has a measured affine restriction. These are selected reference-supplied diagnostics, not autonomous language success.',
                  next_decision='Reject unchanged compact joint adaptation and a sampler-only repair. Evaluate the earlier model with readable image answers on this same fixed cohort before another fit; retain its larger inherited-body cost. Use that control to choose a bounded reader/writer intervention, without treating source-context bypasses as visual language success.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelcolor':'#233441','text.color':'#233441','axes.titleweight':'bold','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),sharey=True)
    for ax,split,title in zip(axes,['train','development'],['a  Seen training records','b  Development records']):
        pos=np.arange(len(ARMS))
        for offset,model,color in [(-.18,'v1','#386985'),(.18,'wider','#A7774D')]:
            vals=[next(r['foreground_f1'] for r in table if r['model']==model and r['split']==split and r['arm']==arm) for arm in ARMS]
            ax.bar(pos+offset,vals,width=.34,color=color,label='V1 · width 384' if model=='v1' else 'Wider · width 1,024')
        ax.set_title(title,loc='left');ax.set_xticks(pos,LABELS,rotation=45,ha='right',fontsize=8)
        ax.set_ylim(0,1.05);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Foreground F1 against supplied target pixels')
    axes[1].legend(frameon=False,loc='upper right',fontsize=8)
    fig.text(.02,.015,'Reference lengths supplied in every writer arm; selected diagnostic samples, not language-answer accuracy. 16 records per split; 122 / 131 target glyphs.',fontsize=8)
    fig.tight_layout(rect=(0,.1,1,1))
    fig.savefig(DEST/'conditional_writing.png',dpi=240);fig.savefig(DEST/'conditional_writing.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for j,model in enumerate(['v1','wider']):
        vals=[]
        for split in ['train','development']:
            rr=[r for r in x['reader_metrics'] if r['model']==model and r['split']==split]
            count=sum(r['native_positions'] for r in rr)
            vals.append([sum(r[k]*r['native_positions'] for r in rr)/count for k in ['affine_relative_mse_floor','achieved_feedback_relative_mse']])
        pos=np.arange(2)
        axes[j].bar(pos-.18,[v[0] for v in vals],.34,color='#AAB8C2',label='Best point in fixed affine family')
        axes[j].bar(pos+.18,[v[1] for v in vals],.34,color='#386985',label='Achieved image-encoder error')
        axes[j].set_xticks(pos,['Training','Development']);axes[j].set_ylim(0,1)
        axes[j].set_title(('a  V1' if model=='v1' else 'b  Wider writer')+' · feedback rank 383',loc='left')
        axes[j].grid(axis='y',alpha=.2);axes[j].set_axisbelow(True)
    axes[0].set_ylabel('Relative squared native-embedding error')
    axes[1].legend(frameon=False,fontsize=7,loc='upper left')
    fig.text(.02,.015,'Uniform selected source positions. Exact-arithmetic affine relaxation of the trained head; no language-quality bound or confidence interval.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(DEST/'reader_alignment.png',dpi=240);fig.savefig(DEST/'reader_alignment.pdf');plt.close(fig)
    public['plot_sha256']={p.name:sha(p) for p in DEST.glob('*') if p.suffix in ['.png','.pdf']}
    payload=json.dumps(public,ensure_ascii=False,indent=2)+'\n'
    if '/home/' in payload:raise ValueError('Private absolute path in public receipt')
    RECEIPT.write_text(payload)
    print(json.dumps({'public_receipt':str(RECEIPT.relative_to(ROOT)),'sha256':sha(RECEIPT),'raw_images':len(public['raw_examples']),
                      'stage_costs':public['accounted_stage_costs'],'first_span':first},indent=2))


if __name__=='__main__':main()
