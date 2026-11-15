#!/usr/bin/env python3
"""Build the completed reward-control revision with preserved native evidence."""
import argparse
import fcntl
from pathlib import Path
import shutil
import build_ranking_completed_article as base

HERE, ROOT = base.HERE, base.ROOT
STAGE = ROOT / 'artifacts/reward_calibration_publication_v1/article_attempt01'
RECEIPT = HERE / 'evidence/reward_calibration_article_build.json'
BASELINE = HERE / 'evidence/glyph_score_gap_article_build.json'
assert base.sha(BASELINE) == '0092069d15a99890e546718dcb172ae4eb628ee4fb04f72e3035fc9b026f663b'
BASE_SHA = base.sha(Path(base.__file__))
assert BASE_SHA == '5470d5963d62bd55d58502c8441538076cbe032178042f7479935a927c8e7d8d'
base.STAGE = STAGE
base.FIGURES = [*base.FIGURES,
    ('Figure_28_glyph_score_gaps', 'glyph_score_gap_caption.tex', 'fig:glyph-score-gaps'),
    ('Figure_29_reward_calibration', 'reward_calibration_caption.tex', 'fig:reward-calibration')]
original_literal = base.literal_figures
def literal_figures():
    return original_literal().replace('build_ranking_completed_article.py', 'build_reward_calibration_article.py')
base.literal_figures = literal_figures


def entrypoint_binding():
    return dict(entrypoint_sha256=base.sha(__file__), reused_builder_sha256=BASE_SHA,
                reused_builder='build_ranking_completed_article.py', figure_plate_count=len(base.FIGURES))


def bound_build():
    value = base.load(STAGE / 'build_receipt.json')
    assert value['revision_entrypoint'] == entrypoint_binding()
    return value


def promote():
    build = bound_build(); inspection = base.load(STAGE / 'inspection.json'); review = base.load(STAGE / 'visual_review.json')
    assert review['complete'] and not review['Office_application_export_verified']
    for name, info in inspection['documents'].items():
        seen = review['documents'][name]
        assert seen['pdf_sha256'] == info['pdf_sha256']
        assert seen['overview_pages'] == list(range(1, info['pages'] + 1)) and seen['detailed_pages']
    for name, digest in build['source_sha256'].items(): assert base.sha(HERE / name) == digest
    base.preserved(); base.model_sources(); protected = build['protected_model_processes']
    for row in protected:
        current = base.identity(row['pid']); assert current['start_ticks'] == row['start_ticks'] and current['state'] != 'Z'
    assert not RECEIPT.exists()
    for folder, name in base.DOCUMENTS:
        source = STAGE / folder / (name + '.pdf'); assert base.sha(source) == inspection['documents'][name]['pdf_sha256']
        target = HERE / (name + '.pdf'); temporary = target.with_suffix('.promoting.pdf')
        shutil.copy2(source, temporary); temporary.replace(target)
    shutil.copy2(STAGE / 'redlines/redlined_manuscript/redlined_manuscript.tex', HERE / 'redlined_manuscript.tex')
    base.model_sources()
    for row in protected:
        current = base.identity(row['pid']); assert current['start_ticks'] == row['start_ticks'] and current['state'] != 'Z'
    build.update(promoted=True, visual_review_complete=True, inspection=inspection, visual_review=review,
        immediate_baseline=base.load(STAGE / 'before_manifest.json'), active_article_updated=True,
        new_optimizer_updates=0, new_model_calls=0, language_of_running_white_target_fit_unmeasured=True,
        builder_with_inspection_sha256=base.sha(__file__), model_ownership_verified_before_and_after_build=True)
    build.pop('protected_model_processes')
    for child in build['costs']['children']:
        child['command']=[('python' if word==base.os.sys.executable else word.replace(str(ROOT)+'/', '')) for word in child['command']]
    build['compiled_previous_redline_source_sha256']=build['source_sha256'].pop('redlined_manuscript.tex')
    build['new_redline_source_sha256']=base.sha(HERE/'redlined_manuscript.tex')
    with RECEIPT.open('x') as stream: base.json.dump(build,stream,indent=2);stream.write('\n')
    print('Promoted the visually reviewed reward-calibration article revision.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-literal',action='store_true');parser.add_argument('--inspect',action='store_true');parser.add_argument('--promote',action='store_true');args=parser.parse_args()
    with (ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.write_literal:
            (HERE/'complete_continuation_figures.tex').write_text(literal_figures())
            print('Integrated literal Figures23--29 with unchanged prior captions and widths.')
        elif args.inspect: bound_build();base.inspect()
        elif args.promote: promote()
        else:
            authority=base.load(ROOT/'artifacts/glyph_tail_current_handoff.json')['runtime_authority']
            bootstrap=authority['active_bootstrap_supervisor']; protected=base.identity(bootstrap['pid'])
            assert protected['start_ticks']==bootstrap['start_ticks'] and protected['state']!='Z'
            base.build();value=base.load(STAGE/'build_receipt.json');value['revision_entrypoint']=entrypoint_binding()
            after=base.identity(protected['pid']);assert after['start_ticks']==protected['start_ticks'] and after['state']!='Z'
            value['protected_model_processes'].append(protected)
            (STAGE/'build_receipt.json').write_text(base.json.dumps(value,indent=2)+'\n')
