#!/usr/bin/env python3
"""Copy the verified, selected paper package and guard the owned latest PDFs."""
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=Path('/home/lachlan/Nutstore Files/Projects/ILM')
NAME='ILM_NatureStyle_20261111_NativeReading'


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old=load(HERE/'evidence/glyph_input_transfer_local_sync.json')
    previous=BASE/old['package_name'];destination=BASE/NAME
    if destination.exists():raise FileExistsError('Preserve existing delivery: '+str(destination))
    build=load(HERE/'evidence/native_reader_writing_publication_build.json')
    if not build['visual_QA']['visual_QA_complete']:raise ValueError('Finish actual visual review before delivery')
    for name,digest in build['files_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Verified current source changed: '+name)
    for name,digest in old['files_sha256'].items():
        if sha(previous/name)!=digest:raise ValueError('Previous package changed: '+name)
    for name,digest in old['latest_alias_sha256'].items():
        if sha(BASE/name)!=digest:raise ValueError('Owned latest alias has a newer unreviewed value: '+name)
    names=set(old['files_sha256'])|set(build['files_sha256'])|{
        'build_native_reader_writing_publication.py','build_native_reader_writing_redline.py',
        'prepare_native_reader_writing_evidence.py','format_native_reader_chart_labels.py',
        'native_reader_writing_figures.mjs','verify_native_reader_writing_figure.py',
        'sync_native_reader_writing_publication.py',
        'evidence/native_reader_writing_publication_build.json',
        'evidence/native_reader_writing_figure_audit.json','evidence/native_reader_writing_figure_bindings.json',
        'evidence/native_reader_writing_redline_build.json'}
    refs=['glyph_tail_native_reader_result.md','glyph_tail_native_reader_protocol.md','glyph_tail_native_reader_decision.md',
        'glyph_tail_first_writing_result.md','glyph_tail_first_writing_protocol.md','glyph_tail_first_writing_decision.md',
        'glyph_tail_context_writer_decision.md','manuscript_native_reader_writing_revision_plan.md',
        'evidence/glyph_tail_native_reader_learning_20261111.json','evidence/glyph_tail_native_reader_registration_20261111.json',
        'evidence/glyph_tail_first_writing_learning_20261111.json','evidence/glyph_tail_first_writing_registration_20261111.json']
    names|={'references/'+n for n in refs}
    names|={str(p.relative_to(ROOT)) for p in (ROOT/'references/evidence/glyph_tail_first_writing_20261111').iterdir() if p.is_file()}
    files={}
    for name in sorted(names):
        if 'baseline_sources/' in name or 'node_modules/' in name:raise ValueError('Generated baseline/cache cannot enter delivery')
        source=ROOT/name if name.startswith('references/') else HERE/name
        if not source.is_file():raise FileNotFoundError(source)
        files[name]=sha(source)
    destination.mkdir()
    for name,digest in files.items():
        source=ROOT/name if name.startswith('references/') else HERE/name
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha(target)!=digest:raise ValueError('Copied file differs: '+name)
    aliases={}
    for name,source in [('ILM_Manuscript_LATEST.pdf',HERE/'manuscript.pdf'),
                        ('ILM_Manuscript_Review_LATEST.pdf',HERE/'manuscript_review.pdf')]:
        if sha(BASE/name)!=old['latest_alias_sha256'][name]:raise ValueError('Alias changed during copy: '+name)
        temporary=BASE/(name+'.native-reading.tmp');shutil.copy2(source,temporary)
        if sha(temporary)!=sha(source):raise ValueError('Staged alias differs')
        temporary.replace(BASE/name);aliases[name]=sha(BASE/name)
    receipt=dict(package_name=NAME,files_sha256=files,previous_owned_alias_sha256=old['latest_alias_sha256'],
        latest_alias_sha256=aliases,local_hashes_verified=True,previous_package_retained=True,
        previous_package_all_manifest_files_reverified=len(old['files_sha256']),
        selected_current_publication_and_bound_references_only=True,archived_source_baselines_excluded=True,
        receipt_self_excluded_from_manifest=True,cloud_propagation_verified=False)
    target=HERE/'evidence/native_reader_writing_local_sync.json'
    target.write_text(json.dumps(receipt,indent=2)+'\n')
    copied=destination/'evidence/native_reader_writing_local_sync.json';shutil.copy2(target,copied)
    if sha(copied)!=sha(target):raise ValueError('Receipt copy differs')
    print(json.dumps(dict(package=NAME,manifest_files=len(files),receipt_verified_separately=True,
        previous_package_files_unchanged=len(old['files_sha256']),owned_latest_aliases_verified=True,
        cloud_propagation_verified=False)))


if __name__=='__main__':main()
