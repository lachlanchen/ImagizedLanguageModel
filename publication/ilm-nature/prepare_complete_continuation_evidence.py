#!/usr/bin/env python3
"""Bind completed language and physical results for native Figure24; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'evidence/complete_continuation_measured_data.json'
SOURCES={
 'references/evidence/visual_continuation_learning_20261113.json':'94c4f77615372f37da9454be2e6cf810dd87d96143148644d1b8a54697db68de',
 'references/evidence/visual_continuation_trajectories_20261113.json':'0996f8277eee0e99e42eeaa6cf91434086ee25c0065f53efff32fc2af2ad99ad',
 'references/evidence/continuation_clean_history_result_20261113.json':'d4f49c5fada27060e6c735b2f0f2eb3729090cd55a6333e87879fb6174d7479f'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())


def build():
 for name,digest in SOURCES.items():assert sha(ROOT/name)==digest,name
 language,trajectory,clean=[load(ROOT/name) for name in SOURCES]
 assert language['generated_streams']==732 and language['reviewed_boards']==183
 assert language['original_pilot_passed'] is False and language['registered_protocol_pilot_passed'] is False
 assert len(clean['records'])==248 and clean['autonomous_language_evidence'] is False
 steps=[0,10324,20649,30973,41298]
 series={}
 for layout in ['fixed_cells','proportional']:
  rows=[r for r in language['core_results'][layout] if r['kind']=='qa' and r['condition']=='normal']
  assert [r['step'] for r in rows]==steps and all(r['cases']==32 for r in rows)
  series[layout]={k:[r[k] for r in rows] for k in ['complete_exact','readable','paired_both_correct','stopped']}
 free=[]
 for step in [0,41298]:
  g=next(x for x in trajectory['aggregate'] if x['arm']=='candidate_long640' and x['step']==step and x['condition']=='normal')
  assert g['streams']==12
  free.append(dict(step=step,glyphs=g['glyphs'],above_clean99=g['above_clean99_count'],
                   fraction=g['above_clean99_count']/g['glyphs']))
 forced=next(x for x in clean['aggregate'] if x['step']==41298 and x['split']=='development' and x['kind']=='book')
 assert forced['cases']==12 and forced['positions']==1161
 books=next(x for x in language['long_book_results'] if x['arm']=='candidate_long640' and x['step']==41298 and x['condition']=='normal')
 assert books['coherent_complete']==0
 retention=language['retention']
 table=[['Final outcome','Fixed cells','Proportional'],
        ['Exact QA /32',str(series['fixed_cells']['complete_exact'][-1]),str(series['proportional']['complete_exact'][-1])],
        ['Readable QA /32',str(series['fixed_cells']['readable'][-1]),str(series['proportional']['readable'][-1])],
        ['Both correct /16',str(series['fixed_cells']['paired_both_correct'][-1]),str(series['proportional']['paired_both_correct'][-1])],
        ['Retained exact',*[f"{len(retention[k]['retained'])}/{len(retention[k]['required'])}" for k in ['fixed_cells','proportional']]],
        ['Coherent books /12','0','Not run']]
 qa=next(x for x in clean['aggregate'] if x['step']==41298 and x['split']=='development' and x['kind']=='qa' and x['layout']=='proportional')
 return dict(schema='complete-continuation-publication-data-v1',source_sha256=SOURCES,
   extractor_sha256=sha(Path(__file__)),steps=steps,qa_series=series,final_table=table,
   book_density=[*free,dict(step=41298,history='supplied',glyphs=forced['positions'],fraction=forced['above_clean99_fraction'])],
   clean_book_ink_99=trajectory['clean_ink_count_quantiles']['book']['0.99'],
   later_proportional_QA=dict(positions=qa['later_positions'],fraction_at_least_095=qa['later_fraction_f1_095'],
     exact=qa['later_exact']),final_long_book_stops=books['stopped'],final_long_book_cases=books['cases'],
   training=language['training'],whole_model_parameters=language['model_parameters'],
   old_language_scores_unchanged=True,all_values_measured=True,original_pilot_passed=False,
   prior_cohort_familiarity=True,independent_human_adjudication=False,new_model_calls=0,
   joint_writer_language_unmeasured=True,active_article_updated=False)


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true');a=p.parse_args()
 payload=json.dumps(build(),ensure_ascii=False,indent=2)+'\n'
 if a.verify:assert OUTPUT.read_text()==payload
 else:
  with OUTPUT.open('x') as f:f.write(payload)
 print(json.dumps(dict(data_sha256=sha(OUTPUT),new_model_calls=0)))
