#!/usr/bin/env python3
"""Preserve both new-context conditions under their actual model identities."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/archive_balanced_glyph_pair.py'


def load(path):return json.loads(Path(path).read_text())


def main(balanced,parent,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only pair archive')
    for name in [OWN,'scripts/archive_glyph_tail_pair.py','scripts/audit_balanced_glyph_writer.py']:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),
                'commit evidence archivers first')
    require(not output.exists(),'preserve earlier pair archives')
    from scripts.archive_glyph_tail_pair import reviewed,archive
    a,b=reviewed(balanced),reviewed(parent)
    for run,other in [(balanced,parent),(parent,balanced)]:
        require((ROOT/load(run/'comparison_registration.json')['companion_run']).resolve()==other,
                'archive both registered companions')
    require({r['key']:r['input_sha256'] for r in a[0]['records']}
            =={r['key']:r['input_sha256'] for r in b[0]['records']},'comparison physical inputs differ')
    m=load(balanced/'preparation.json');pm=load(parent/'preparation.json')
    require(m['balanced_writer_refinement'] and pm['unchanged_parent_baseline']
            and m['fresh_cases']==pm['fresh_cases'],'new trial condition identity differs')
    previous=load(ROOT/m['comparison_parent']/'preparation.json')
    require({r['context_hash'] for r in m['fresh_cases']}.isdisjoint(r['context_hash'] for r in previous['fresh_cases']),
            'do not recycle the earlier reserved contexts as a fresh test')
    audit=load(balanced/'final_checkpoint_audit.json')
    require(audit['checkpoint_sha256']==sha(balanced/'fit/checkpoint_latest.pt')
            and audit['inherited_checkpoint_sha256']==sha(parent/'fit/checkpoint_latest.pt')
            and audit['inherited_body_adapters_bit_exact'],'independent final tensor audit required')
    output.mkdir(parents=True)
    conditions={name:archive(run,output/name,data) for name,run,data in
        [('balanced_writer',balanced,a),('unchanged_parent',parent,b)]}
    for name in ['final_checkpoint_audit.json','cache/report.json','input_review.json']:
        source=balanced/name;target=output/('balanced_'+name.replace('/','_'))
        shutil.copy2(source,target)
    for run,label in [(balanced,'balanced_writer'),(parent,'unchanged_parent')]:
        source=run/'raw_review/inspection_zooms'
        if source.exists():shutil.copytree(source,output/label/'inspection_zooms')
    license_path=ROOT/'data/visual_qa/cmrc2018_short/LICENCE';shutil.copy2(license_path,output/'CMRC_SOURCE_LICENCE')
    save_json(output/'archive.json',dict(archiver_sha256=sha(ROOT/OWN),
        common_archiver_sha256=sha(ROOT/'scripts/archive_glyph_tail_pair.py'),conditions=conditions,
        total_streams=144,both_conditions_retained=True,all_52_actual_review_boards_retained=True,
        comparison_is_balanced_writer_vs_same_unchanged_contextual_parent=True,
        previous_first_reserved_contexts_reused=False,
        parent_physical_qualification_reused_for_identical_weights=True,
        refinement_inherited_full_parent_training=True,full_goal_achieved=False,
        source_dataset_license_sha256=sha(license_path),
        extra_files_sha256={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()
            and not any(p.is_relative_to(output/n) for n in conditions)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--balanced',type=Path,default=Path('artifacts/balanced_glyph_writer_v1'))
    p.add_argument('--parent',type=Path,default=Path('artifacts/balanced_glyph_parent_v1'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.balanced.resolve(),a.parent.resolve(),a.output.resolve())
