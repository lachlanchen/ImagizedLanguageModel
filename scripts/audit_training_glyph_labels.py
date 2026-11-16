#!/usr/bin/env python3
"""Register/run full training-only source-to-cell supervision verification."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

from ilm.training_only.glyph_label_audit import add_pairs, inventory_summary, verify_record
from scripts.prepare_visual_sequence_corpus import ROOT, FONT, Rasterizer, documents, text_hash

CORPUS = ROOT / 'data/visual_sequence_corpus_v1'
RUN = ROOT / 'artifacts/training_glyph_label_audit_v1'
REG = ROOT / 'references/evidence/training_glyph_label_registration_20261116.json'
OUT = ROOT / 'references/evidence/training_glyph_label_result_20261116.json'
PROTOCOL = ROOT / 'references/training_glyph_label_protocol_20261116.md'


def load(path): return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream: json.dump(value,stream,ensure_ascii=False,indent=2); stream.write('\n')


def guard(began):
    if time.monotonic() - began > 900: raise RuntimeError('900-second CPU ceiling exceeded')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2 > 2: raise RuntimeError('2 GiB RSS ceiling exceeded')
    mem = next(v for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
    if int(mem.split()[1]) / 1024**2 < 24: raise RuntimeError('24 GiB host reserve exhausted')
    fs = os.statvfs(ROOT)
    if fs.f_bavail * fs.f_frsize / 1024**3 < 48: raise RuntimeError('48 GiB disk reserve exhausted')


def contract():
    report = load(CORPUS / 'report.json')
    if sha(ROOT / 'scripts/prepare_visual_sequence_corpus.py') != report['source_code_sha256']:
        raise ValueError('Original renderer/source selector changed')
    if sha(FONT) != report['font_sha256'] or report['font_face'] != 2:
        raise ValueError('Original font identity changed')
    bindings = dict(report['source_sha256'])
    for name, digest in bindings.items():
        if sha(ROOT / name) != digest: raise ValueError('Original raw source changed: ' + name)
    for path in [Path(__file__), PROTOCOL, CORPUS / 'report.json', CORPUS / 'verification.json',
                 ROOT / 'ilm/training_only/glyph_label_audit.py', ROOT / 'tests/test_glyph_label_audit.py',
                 ROOT / 'scripts/prepare_visual_sequence_corpus.py', ROOT / 'ilm/image_edit/baseline.py',
                 ROOT / 'data/visual_qa/natural_first_images_v1/report.json']:
        bindings[str(path.relative_to(ROOT))] = sha(path)
    shards = []
    for shard in report['shards']:
        if shard['split'] != 'train': continue
        files = {}
        for suffix in ('_sources.json', '_pixels.npz'):
            name, = [n for n in shard['files'] if n.endswith(suffix)]
            if Path(name).name != name: raise ValueError('Invalid training shard path')
            file = CORPUS / name
            if sha(file) != shard['files'][name]: raise ValueError('Training source identity differs')
            files[suffix] = str(file.relative_to(ROOT)); bindings[files[suffix]] = sha(file)
        shards.append(dict(records=shard['records'], glyphs=shard['glyphs'], files=files))
    if sum(v['records'] for v in shards) != 13785: raise ValueError('Keep all original training records')
    return bindings, shards, report


def register():
    bindings, shards, report = contract()
    write_new(REG, dict(schema='training-glyph-label-registration-v1', input_sha256=bindings,
        font_file=str(FONT), font_sha256=report['font_sha256'], font_face=2, training_shards=shards,
        expected_records=13785, expected_glyphs=sum(v['glyphs'] for v in shards),
        expected_target_glyphs=sum(v for k,v in report['totals'].items() if k.startswith('train/') and k.endswith('/target_glyphs')),
        checks=['every training record source/group/text hash and prompt boundary authentic',
                'all original training cells byte-equal original renderer',
                'every nonwhite physical mask has exactly one raw canonical character',
                'physically white cells have only whitespace source labels',
                'all records/positions retained; no development image or label added'],
        resources=dict(cpu_threads=2, maximum_seconds=900, maximum_rss_gib=2, minimum_available_host_gib=24,
                       minimum_free_disk_gib=48, maximum_derived_data_gib=1),
        action_on_failure='Reject exact context-free raw-character supervision under this rendering; preserve every alias/mismatch. No verifier or actor fit.',
        action_on_pass='Source/label support only; separately register verifier training, corruption, abstention and complete calibration before any actor preflight.',
        neural_calls=0, optimizer_updates=0, verifier_training_authorized=False, actor_training_authorized=False))
    print(json.dumps(dict(registered=str(REG.relative_to(ROOT)), shards=len(shards), records=13785)))


def run():
    began = time.monotonic(); guard(began)
    reg = load(REG); bindings, shards, report = contract()
    if reg['input_sha256'] != bindings or reg['training_shards'] != shards:
        raise ValueError('Registered supervision audit changed')
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True,timeout=30).split()[0]
    committed = subprocess.check_output(['git','show','HEAD:'+str(REG.relative_to(ROOT))],cwd=ROOT)
    if head != remote or hashlib.sha256(committed).hexdigest() != sha(REG):
        raise ValueError('Commit and push the audit before deriving labels')
    if RUN.exists() or OUT.exists(): raise FileExistsError('Preserve prior audit; no automatic retry')
    RUN.mkdir(parents=True)
    write_new(RUN / 'owner.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), started_unix=time.time(),registration_sha256=sha(REG)))
    verification = load(CORPUS/'verification.json')
    if verification['report_sha256'] != sha(CORPUS/'report.json') or verification['cross_partition_exact_input_or_source_group_overlap'] != 0:
        raise ValueError('Require authenticated group-disjoint original corpus')
    try:
        source = {}
        for split,kind,identifier,group,prompt,answer in documents():
            if split != 'train': continue
            key = (kind,identifier)
            if key in source: raise ValueError('Duplicate raw training identity')
            source[key] = (group,prompt,answer)
        bank = {}; renderer = Rasterizer(); records = []; labels = []; offsets = [0]
        mismatch_positions = target_positions = 0; counts = Counter()
        for number, shard in enumerate(shards,1):
            guard(began)
            origins = load(ROOT / shard['files']['_sources.json'])
            with np.load(ROOT / shard['files']['_pixels.npz'],allow_pickle=False) as data:
                ink = data['ink']; bounds = data['record_glyph_offsets']; prompt_lengths = data['prompt_glyphs']
            if len(origins) != shard['records'] or len(ink) != shard['glyphs'] or len(bounds) != len(origins)+1:
                raise ValueError('Complete shard extent differs')
            if bounds[0] != 0 or bounds[-1] != len(ink) or np.any(np.diff(bounds)<1):
                raise ValueError('Invalid complete record partition')
            for local, origin in enumerate(origins):
                guard(began)
                key = (origin['kind'],origin['id'])
                group,prompt,answer = source[key]; text = prompt+answer
                if group != origin['group'] or text_hash(text) != origin['source_text_sha256']:
                    raise ValueError('Original source text/group differs from actual corpus')
                first,last = bounds[local:local+2]; packed = ink[first:last]
                bad = verify_record(packed,text,len(prompt),int(prompt_lengths[local]),renderer)
                mismatch_positions += bad; target_positions += len(answer)
                add_pairs(bank,packed,text)
                labels.append(np.fromiter(map(ord,text),dtype=np.uint32,count=len(text)))
                offsets.append(offsets[-1]+len(text)); counts[origin['kind']] += 1
                records.append(dict(**origin,corpus_index=len(records),glyphs=len(text),prompt_glyphs=len(prompt),
                                    target_glyphs=len(answer),renderer_mismatch_positions=bad))
            print(json.dumps(dict(phase='training_label_audit',shards=number,total_shards=len(shards),
                                  records=len(records),positions=offsets[-1],physical_masks=len(bank))),flush=True)
        summary = inventory_summary(bank)
        if len(records) != reg['expected_records'] or offsets[-1] != reg['expected_glyphs'] or target_positions != reg['expected_target_glyphs']:
            raise ValueError('Keep every scheduled original source position')
        bank_offsets = [0]; codepoints=[]; occurrences=[]
        ordered = sorted(bank)
        for mask in ordered:
            for cp,count in sorted(bank[mask].items()): codepoints.append(cp);occurrences.append(count)
            bank_offsets.append(len(codepoints))
        np.savez_compressed(RUN/'physical_labels.npz',ink=np.frombuffer(b''.join(ordered),dtype=np.uint8).reshape(-1,128),
            label_offsets=np.asarray(bank_offsets,dtype=np.int64),codepoints=np.asarray(codepoints,dtype=np.uint32),
            occurrences=np.asarray(occurrences,dtype=np.int64))
        np.savez_compressed(RUN/'record_labels.npz',codepoints=np.concatenate(labels),
            record_offsets=np.asarray(offsets,dtype=np.int64),prompt_glyphs=np.asarray([r['prompt_glyphs'] for r in records],dtype=np.int32))
        write_new(RUN/'records.json',records)
        guard(began)
        files = {str(p.relative_to(ROOT)):sha(p) for p in [RUN/'physical_labels.npz',RUN/'record_labels.npz',RUN/'records.json']}
        size = sum((ROOT/p).stat().st_size for p in files)
        if size > 1024**3: raise RuntimeError('Derived label data exceeds 1 GiB')
        checks = dict(all_records_authenticated=True,all_original_cells_exactly_rendered=mismatch_positions==0,
            all_nonwhite_masks_have_unique_raw_character=summary['context_free_nonwhite_labels_unique'],
            physically_white_cells_only_whitespace=summary['physically_white_labels_only_whitespace'],
            full_training_extent_retained=True,development_images_or_labels_added=False)
        passed = all(v for k,v in checks.items() if k!='development_images_or_labels_added') and not checks['development_images_or_labels_added']
        write_new(OUT,dict(schema='training-glyph-label-result-v1',registration_sha256=sha(REG),records=len(records),
            kinds=dict(counts),target_positions=target_positions,**summary,renderer_mismatch_positions=mismatch_positions,
            checks=checks,exact_context_free_supervision_supported=passed,label_files_sha256=files,
            derived_bytes=size,elapsed_seconds=time.monotonic()-began,
            peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            source_classes_not_semantic_understanding=True,neural_calls=0,optimizer_updates=0,
            verifier_training_authorized=False,actor_training_authorized=False,goal_achieved=False))
        print(json.dumps(dict(complete=True,exact_context_free_supervision_supported=passed,
            nonwhite_aliases=len(summary['nonwhite_aliases']),white_nonspace_labels=len(summary['white_nonspace_labels']),
            records=len(records),positions=offsets[-1],renderer_mismatches=mismatch_positions)),flush=True)
    except BaseException as error:
        write_new(RUN/'failure.json',dict(error_type=type(error).__name__,error=str(error),
            elapsed_seconds=time.monotonic()-began,automatic_retry=False,partial_data_preserved=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'));args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('OPENBLAS_NUM_THREADS')!='2' or os.environ.get('OMP_NUM_THREADS')!='2':
        raise ValueError('Require CPU-only two-thread execution')
    if args.action=='register': register()
    else:
        with (ROOT/'artifacts/training_glyph_label_audit.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run()
