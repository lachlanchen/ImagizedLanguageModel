"""Native S42: a passed physical gate without new-context answer gain."""
from PIL import Image
from glyph_step_data import ROOT,EVIDENCE,load,require,sha

FIGURE='Figure_S42_balanced_writer_transfer'


def source_data():
    base=EVIDENCE/'balanced_glyph_writer_pair_v1';bound={};conditions={};cells=[]
    def read(path):bound[str(path.relative_to(ROOT))]=sha(path);return load(path)
    archive=read(base/'archive.json');require(archive['total_streams']==144,'retain the full new paired cohort')
    for name in ['unchanged_parent','balanced_writer']:
        for path,digest in archive['conditions'][name]['files_sha256'].items():
            p=base/name/path;require(sha(p)==digest,'archived writer comparison changed');bound[str(p.relative_to(ROOT))]=digest
        scores=read(base/name/'scores.json');q=read(base/name/'qualification/report.json')
        require(not scores['pilot_complete_answer_requirement_passed'],'retain failed new-context pilot')
        conditions[name]=dict(groups=scores['groups'],qualification=q,
            paired_contexts=scores['fresh_both_questions_correct_contexts'])
        for index in range(8):
            row,=[r for r in scores['records'] if r['key']==f'fresh/{index:04}/normal']
            path=base/name/'raw_outputs'/row['png']
            require(sha(path)==row['png_sha256'] and Image.open(path).size==(max(1,row['glyphs'])*32,32),
                    'fixed actual refinement output differs')
            cells.append(dict(condition=name,index=index,file=str(path.relative_to(ROOT)),sha256=sha(path),
                glyphs=row['glyphs'],stopped=row['stopped'],capped=row['capped'],
                displayed_glyphs=min(10,row['glyphs']),display_is_prefix_excerpt=row['glyphs']>10))
    audit=read(base/'balanced_final_checkpoint_audit.json');cache=read(base/'balanced_cache_report.json')
    positions=read(base/'training_position_summary.json')
    require(audit['inherited_body_adapters_bit_exact'] and cache['actual_parent_training_pixels_and_stop_counts_reproduced'],
            'actual unchanged-body evidence required')
    require([conditions[n]['groups']['fresh/normal']['manual_clear_complete_exact'] for n in conditions]==[19,18]
        and conditions['balanced_writer']['qualification']['readout_gate_passed'],'measured refinement outcome differs')
    return dict(conditions=conditions,cells=cells,source_files_sha256=bound,training_positions=positions,
        all144_outputs_preserved=True,fixed_example_indices=list(range(8)),new_contexts_differ_from_first_trial=True,
        usable_language_established=False)


def append(paper):
    from contextual_answer_figures import append as common_append
    return common_append(paper,balanced=True)
