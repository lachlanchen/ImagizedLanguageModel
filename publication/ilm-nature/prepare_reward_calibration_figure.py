#!/usr/bin/env python3
"""Export the two terminal reward controls for a native manuscript figure."""
import fcntl
import json
from pathlib import Path
import shutil
from PIL import Image
from prepare_glyph_score_gap_figure import ROOT, HERE, sha, load, owners
from build_ranking_completed_article import model_sources, host


def main():
    protected = owners(); model_sources(); assert host()['available_gib'] >= 48
    evidence = ROOT / 'references/evidence'
    names = ['whole_answer_reward_calibration_20261116.json',
             'whole_answer_reward_calibration_registration_20261116.json',
             'selective_visual_verifier_calibration_20261116.json',
             'selective_visual_verifier_registration_20261116.json']
    paths = [evidence / n for n in names]; a, ar, b, br = map(load, paths)
    assert a['registration_sha256'] == sha(paths[1]) and b['registration_sha256'] == sha(paths[3])
    assert not a['calibration_passed'] and not b['calibration_passed']
    assert a['new_model_calls'] == b['neural_calls'] == a['optimizer_updates'] == b['optimizer_updates'] == 0
    rows_a = {(r['layout'], r['case']): r for r in a['rows']}
    rows_b = {(r['layout'], r['case']): r for r in b['rows']}
    assert len(rows_a) == len(rows_b) == 64 and rows_a.keys() == rows_b.keys()
    strata = [('Correct', lambda r: r['complete_exact']),
              ('Readable wrong', lambda r: r['readable'] and not r['complete_exact']),
              ('Uncredited / uncertain', lambda r: not r['readable'])]
    groups = []
    for label, condition in strata:
        aa = [r for r in a['rows'] if condition(r)]; bb = [r for r in b['rows'] if condition(r)]
        assert len(aa) == len(bb)
        groups.append(dict(label=label, cases=len(aa), selective_accepted=sum(r['complete_content'] for r in bb)))
    assert [r['cases'] for r in groups] == [9, 9, 46]
    assert [r['selective_accepted'] for r in groups] == [7, 0, 1]
    extrema = []
    for label, result in [('Image edit', a), ('Selective verifier', b)]:
        groups_r = [[r['reward'] for r in result['rows'] if cond(r)] for _, cond in strata]
        values = [min(groups_r[0]), max(groups_r[1]), max(groups_r[2])]
        extrema.append(dict(name=label, exact_values=values, values=[round(v, 6) for v in values]))
    samples = []
    selected = [r for r in b['rows'] if (r['complete_exact'] and not r['complete_content'])
                or (not r['readable'] and r['complete_content'])]
    assert len(selected) == 3
    folder = HERE / 'figures/reward_calibration_samples'; folder.mkdir(exist_ok=False)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    for r in sorted(selected, key=lambda v: (not v['complete_exact'], v['layout'], v['case'])):
        source = ROOT / 'artifacts/visual_continuation_full_logger_recovery_v2/evaluation/step_041298/core128' / (r['layout'] + '__' + r['case']) / 'answer.png'
        key = str(source.relative_to(ROOT)); assert sha(source) == br['input_sha256'][key]
        target = folder / (r['layout'] + '__' + r['case'] + '.png'); shutil.copy2(source, target)
        assert sha(target) == sha(source); sources[key] = sha(source)
        with Image.open(target) as image: width, height = image.size
        samples.append(dict(case=r['case'], layout=r['layout'], file=str(target.relative_to(ROOT)),
            sha256=sha(target), width=width, height=height, accepted_cells=r['accepted_cells'],
            abstained_cells=r['abstained_cells'], emitted_glyphs=r['emitted_glyphs'],
            frozen_correct=r['complete_exact'], frozen_uncertain=r['uncertain']))
    out = dict(schema='reward-calibration-figure-data-v1', source_sha256=sources,
        groups=groups, extrema=extrema, samples=samples,
        categories=['Minimum\ncorrect', 'Maximum\nreadable wrong', 'Maximum\nuncredited'],
        chart_decimals=6, image_edit_auc=a['correct_vs_readable_wrong_auc'], selective_auc=b['correct_wrong_auc'],
        selective_inventory=b['inventory'], resource_scopes=dict(image=a['elapsed_scoring_seconds'],
            selective=b['elapsed_seconds'], imports_and_source_hashing_excluded=True),
        both_calibrations_failed=True, original_readings_unchanged=True,
        independent_human_adjudication=False, student_training=False, new_model_calls=0,
        exporter_sha256=sha(__file__))
    model_sources(); assert owners() == protected
    with (HERE / 'evidence/reward_calibration_figure_data.json').open('x') as stream:
        json.dump(out, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(cases=64, samples=3, groups=groups, new_model_calls=0)))


if __name__ == '__main__':
    with (ROOT / 'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); main()
