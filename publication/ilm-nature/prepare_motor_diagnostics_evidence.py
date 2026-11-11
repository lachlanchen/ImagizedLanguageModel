#!/usr/bin/env python3
"""Recount paired writing controls and export original 32-square figure tiles."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE = Path(os.environ.get('ILM_MOTOR_PAPER_STAGE', str(ROOT/'artifacts/motor_diagnostics_publication_v1')))
EVIDENCE = ROOT/'references/evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def freeze_preserved_assets():
    baseline = {}
    revision = '65f4d9fd'
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision,
        '--', 'publication/ilm-nature'], cwd=ROOT, text=True).splitlines()
    for name in names:
        relative = str(Path(name).relative_to('publication/ilm-nature'))
        if (relative.startswith('figures/') or relative == 'nature_reading.sty'
                or relative.startswith('supplementary_information')
                or relative.startswith('redlined_supplementary')):
            raw = subprocess.check_output(['git', 'show', revision+':'+name], cwd=ROOT)
            if raw.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                baseline[relative] = next(line.split('sha256:', 1)[1]
                    for line in raw.decode().splitlines() if line.startswith('oid sha256:'))
            else:
                baseline[relative] = hashlib.sha256(raw).hexdigest()
            if sha(HERE/relative) != baseline[relative]:
                raise ValueError('Earlier figure/style/supplement changed: '+relative)
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE/'preserved_assets.json').write_text(json.dumps(dict(
        baseline_commit=revision, LFS_content_identity_used=True, files_sha256=baseline), indent=2)+'\n')
    return baseline


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    folder = STAGE/'first_tiles'
    folder.mkdir(exist_ok=True)
    paths = {k:EVIDENCE/n for k,n in dict(
        language='glyph_context_motor_learning_20261111.json',
        capacity='glyph_context_motor_capacity_learning_20261111.json',
        decoded='decoder_aware_motor_learning_20261111.json',
        parent='glyph_tail_first_writing_learning_20261111.json').items()}
    language, capacity, decoded, parent = (load(paths[k]) for k in paths)
    assert language['independent_per_answer_and_aggregate_recount_passed']
    assert language['all216_initial_context_states_bit_exact_to_matched_parent']
    assert not decoded['full_language_generation_run']
    assert decoded['whole_language_scores_unmeasured_for_new_model']
    layouts = ['fixed_cells', 'proportional']
    qa = {'parent': {}, 'flow': {}}
    # Parent per-answer scores are authenticated in the first-writing record.
    for layout in layouts:
        rows = [r['job'] for r in parent['records'] if r['job']['layout']==layout
                and r['job']['condition']=='normal']
        pairs = {}
        for r in rows:
            pairs.setdefault(r['target_case'].split('_')[1], []).append(r)
        assert len(rows)==32 and len(pairs)==16 and all(len(p)==2 for p in pairs.values())
        qa['parent'][layout] = dict(exact=sum(r['glyph_complete_exact'] for r in rows),
            pairs=sum(all(r['glyph_complete_exact'] for r in pair) for pair in pairs.values()))
        row = next(r for r in language['results'][layout] if r['kind']=='qa' and r['condition']=='normal')
        qa['flow'][layout] = dict(exact=row['complete_exact'], pairs=row['paired_both_correct'])
    assert qa == {'parent': {'fixed_cells': {'exact':2,'pairs':1}, 'proportional': {'exact':6,'pairs':0}},
                  'flow': {'fixed_cells': {'exact':2,'pairs':1}, 'proportional': {'exact':7,'pairs':1}}}
    archives = {k:EVIDENCE/n for k,n in dict(
        capacity='glyph_context_motor_capacity_20261111/pixels.npz',
        decoded='decoder_aware_motor_20261111/pixels.npz').items()}
    with np.load(archives['capacity'], allow_pickle=False) as old, np.load(archives['decoded'], allow_pickle=False) as new:
        for key in ['gold_ink', 'parent_ink', 'supplied_ink']:
            assert np.array_equal(old[key], new[key])
        assert np.array_equal(old['candidate_ink'], new['control_ink'])
        assert [r['id'] for r in capacity['records']] == [r['id'] for r in decoded['records']]
        # Independent integer recount includes all128 positions for all four routes.
        arrays = dict(parent=new['parent_ink'], flow=new['control_ink'],
                      decoded=new['candidate_ink'], codec=new['supplied_ink'])
        scores = {}
        for group in ['train_first','train_later','fixed_cells','proportional']:
            idx = [i for i,r in enumerate(decoded['records']) if r['group']==group]
            assert len(idx)==32
            scores[group] = {}
            for route,predicted in arrays.items():
                fs = []; exact = 0
                for i in idx:
                    p,g = predicted[i].astype(bool), new['gold_ink'][i].astype(bool)
                    tp = int((p&g).sum()); fp = int((p&~g).sum()); fn = int((~p&g).sum())
                    f = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
                    source = decoded['records'][i]['scores'][dict(flow='control',decoded='candidate').get(route,route)]
                    assert (tp,fp,fn)==(source['tp'],source['fp'],source['fn'])
                    assert f==source['foreground_f1']
                    fs.append(f);exact += int(np.array_equal(p,g))
                s = dict(cases=32, exact=exact, mean_foreground_f1=float(np.mean(fs)),
                         median_foreground_f1=float(np.median(fs)), at_least_095=sum(f>=.95 for f in fs))
                assert s == decoded['groups'][group][dict(flow='control',decoded='candidate').get(route,route)]
                scores[group][route] = s
        pictures=[]
        for case in ['first_02321','later_15112','proportional__qa_02_b_normal','proportional__qa_09_a_normal']:
            i = next(i for i,r in enumerate(decoded['records']) if r['id']==case)
            for kind,array in {'target':new['gold_ink'][i], **{k:v[i] for k,v in arrays.items()}}.items():
                assert array.shape==(1,32,32) and set(np.unique(array))<= {0,1}
                gray=255*(1-array[0].astype(np.uint8))
                target=folder/f'{case}_{kind}.png';Image.fromarray(gray).save(target)
                assert np.array_equal(np.asarray(Image.open(target)),gray)
                pictures.append(dict(case=case,kind=kind,file=str(target.relative_to(STAGE)),
                    sha256=sha(target),gray_pixels_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
                    binary_ink_sha256=hashlib.sha256(array.tobytes()).hexdigest(),width=32,height=32))
    data=dict(source_sha256={k:sha(v) for k,v in paths.items()},
        raw_pixel_archives_sha256={k:sha(v) for k,v in archives.items()},qa=qa,physical=scores,pictures=pictures,
        full_precision_F1_retained=True,chart_F1_decimal_places=6,all128_positions_independently_recounted=True,
        excerpts_retrospectively_selected=True,language_score_for_decoded_model=None,
        unmeasured_language_score_not_zero=True,new_optimizer_updates=0,
        all216_original_flow_streams_preserved=True,all16_physical_boards_preserved=True)
    (STAGE/'measured_data.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(charts=4,original_tiles=len(pictures),preserved_assets=len(freeze_preserved_assets()),
        physical_positions=128,independent_integer_recount=True)))


if __name__=='__main__':
    main()
