#!/usr/bin/env python3
"""Recount completed native/pixel controls and extract unrepaired first tiles."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE = ROOT/'artifacts/native_reader_writing_publication_v2'
EVIDENCE = ROOT/'references/evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def freeze_preserved_assets():
    baseline={};pointers={}
    names=subprocess.check_output(['git','ls-tree','-r','--name-only','75080984','--','publication/ilm-nature'],cwd=ROOT,text=True).splitlines()
    for name in names:
        relative=str(Path(name).relative_to('publication/ilm-nature'))
        if (relative.startswith('figures/') or relative=='nature_reading.sty'
                or relative.startswith('supplementary_information') or relative.startswith('redlined_supplementary')):
            raw=subprocess.check_output(['git','show','75080984:'+name],cwd=ROOT)
            if raw.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                lines=raw.decode().splitlines();oid=next(x.split('sha256:',1)[1] for x in lines if x.startswith('oid sha256:'))
                baseline[relative]=oid
                pointers[relative]=dict(pointer_sha256=hashlib.sha256(raw).hexdigest(),content_sha256=oid,
                    size=int(next(x.split()[1] for x in lines if x.startswith('size '))))
            else:baseline[relative]=hashlib.sha256(raw).hexdigest()
    (STAGE/'preserved_assets.json').write_text(json.dumps(dict(baseline_commit='75080984',
        files_sha256=baseline,large_file_pointer_bindings=pointers),indent=2)+'\n')
    return baseline


def main():
    STAGE.mkdir(exist_ok=True)
    folder = STAGE/'first_tiles'; folder.mkdir(exist_ok=False)
    paths = {k:EVIDENCE/n for k,n in dict(
        native='glyph_tail_native_reader_learning_20261111.json',
        writing='glyph_tail_first_writing_learning_20261111.json',
        registration='glyph_tail_first_writing_registration_20261111.json').items()}
    native, writing, registration = (load(paths[k]) for k in ['native','writing','registration'])
    assert native['all448_cases_authenticated'] and native['actual_adapted_image_prefills_bit_exact'] == 192
    assert writing['independent_integer_pixel_recount_passed'] and len(writing['records']) == 192
    assert sha(paths['registration']) == writing['registration_sha256']
    raw = EVIDENCE/'glyph_tail_first_writing_20261111/pixels.npz'
    assert sha(raw) == writing['public_files_sha256']['pixels.npz']
    normal = [r for r in native['records'] if r['condition']=='normal']
    def count(tail,layout):
        rows = [r for r in normal if r['tail']==tail and r['layout']==layout]
        assert len(rows) == 32
        pairs = defaultdict(list)
        for r in rows:
            pairs[r['paired_group']].append(r)
        assert len(pairs) == 16 and all(len(x)==2 for x in pairs.values())
        both = sum(all(r['complete_exact'] for r in p) and p[0]['text'] != p[1]['text'] for p in pairs.values())
        return dict(exact=sum(r['complete_exact'] for r in rows),pairs=both)
    routes = {tail:{layout:count(tail,layout) for layout in ['fixed_cells','proportional','text']}
              for tail in ['adapted','source_zero_output']}
    jobs = [r for r in writing['records'] if r['job']['condition']=='normal']
    glyph = {}
    for layout in ['fixed_cells','proportional']:
        rows = [r['job'] for r in jobs if r['job']['layout']==layout]
        pairs = defaultdict(list)
        for r in rows:
            pairs[r['target_case'].split('_')[1]].append(r)
        assert len(rows)==32 and len(pairs)==16 and all(len(p)==2 for p in pairs.values())
        glyph[layout] = dict(exact=sum(r['glyph_complete_exact'] for r in rows),
            pairs=sum(all(r['glyph_complete_exact'] for r in p) for p in pairs.values()))
    outcomes = {k:0 for k in ['Native only','Glyph only','Both','Neither']}
    for row in jobs:
        r = row['job']
        if r['layout']!='proportional':continue
        a,b=r['native_complete_exact'],r['glyph_complete_exact']
        outcomes['Both' if a and b else 'Native only' if a else 'Glyph only' if b else 'Neither'] += 1
    assert list(outcomes.values())==[7,3,3,19]
    assert [[routes[t][l]['exact'] for l in ['fixed_cells','proportional']]
            for t in ['adapted','source_zero_output']]==[[4,10],[2,13]]
    assert glyph==dict(fixed_cells=dict(exact=2,pairs=1),proportional=dict(exact=6,pairs=0))
    erased = [r for r in native['records'] if r['condition']!='normal']
    assert len(erased)==256 and not any(r['complete_exact'] for r in erased)
    first = [writing['supplied_first_32']['mean_foreground_f1']]+[
        next(r['mean_foreground_f1'] for r in writing['original_first_aggregates']
             if r['layout']==layout and r['condition']=='normal') for layout in ['fixed_cells','proportional']]
    pictures=[]; targets={r['case']:r for r in registration['targets']}
    indexes={r['id']:i for i,r in enumerate(registration['jobs'])}
    with np.load(raw,allow_pickle=False) as pixels:
        for case in ['qa_09_a','qa_05_b','qa_13_a']:
            offset=targets[case]['offset']
            arrays=[pixels['gold_ink'][offset],pixels['supplied_reconstruction_ink'][offset],
                    pixels['original_first_ink'][indexes['fixed_cells__'+case+'_normal']],
                    pixels['original_first_ink'][indexes['proportional__'+case+'_normal']]]
            for kind,array in zip(['canonical','supplied','fixed','proportional'],arrays):
                gray=(255*(1-array[0])).astype(np.uint8)
                path=folder/f'{case}_{kind}.png';Image.fromarray(gray).save(path)
                assert np.array_equal(np.asarray(Image.open(path)),gray)
                pictures.append(dict(case=case,kind=kind,file=str(path.relative_to(STAGE)),sha256=sha(path),
                    original_binary_pixels_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                    gray_pixels_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),width=32,height=32))
    data=dict(source_sha256={k:sha(v) for k,v in paths.items()},raw_pixel_archive_sha256=sha(raw),
        native=routes,glyph=glyph,proportional_outcomes=outcomes,mean_first_F1=first,
        first_pixel_exact=[31,0,3],pictures=pictures,raw_first_tile_examples_retrospective=True,
        first_pixel_mismatch_not_semantic_error=True,all448_native_calls_preserved=True,
        all192_glyph_streams_preserved=True,new_optimizer_updates=0,new_ILM_images=0)
    (STAGE/'measured_data.json').write_text(json.dumps(data,indent=2)+'\n')
    baseline=freeze_preserved_assets()
    print(json.dumps(dict(charts=4,original_tiles=12,preserved_assets=len(baseline),native_calls=448,first_streams=192)))


if __name__=='__main__':
    if sys.argv[1:]==['--baseline-only']:
        print(json.dumps(dict(preserved_assets=len(freeze_preserved_assets()),LFS_content_identity_used=True)))
    else:main()
