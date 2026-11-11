#!/usr/bin/env python3
"""Export the completed wider-writer review, measurements and fixed raw examples."""

import hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads((R/p).read_text())
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
a=Path('artifacts/compact_visual_span_full_rank_learning_v1');p=Path('artifacts/compact_visual_span_full_rank_pipeline_v1');b=Path('artifacts/compact_visual_span_full_rank_blind_review_v1');e=Path('references/evidence')
x=load(a/'summary.json');scores=load(p/'review/scores.json');frozen=load(b/'readings_frozen.json');fit=load('artifacts/compact_visual_span_full_rank_fit_v1/report.json')
assert frozen['readings_sha256']==sha(b/'readings.json')==scores['readings_sha256']==x['readings_sha256']
assert frozen['images_reviewed']==372 and frozen['frozen_before_reference_scoring']
out=e/'compact_visual_span_full_rank_learning_20261111';out.mkdir(exist_ok=True)
shutil.copytree(R/a/'raw_examples',R/out/'raw_examples',dirs_exist_ok=True)
shutil.copyfile(R/a/'language_metrics.csv',R/out/'language_metrics.csv')
shutil.copyfile(R/b/'readings.json',R/e/'compact_visual_span_full_rank_readings_20261111.json')
for row in x['raw_examples']:assert sha(out/row['file'])==row['image_sha256']
normal=[];resources=[]
for step in [0,1956,4049,6152,8172]:
 rows=[r for r in scores['records'] if r['step']==step and r['condition']=='normal']
 assert len(rows)==44
 normal.append(dict(step=step,cases=44,empty=sum(r['emitted_glyphs']==0 for r in rows),stopped=sum(r['stopped'] for r in rows),capped=sum(not r['stopped'] for r in rows),unique_pngs=len({r['image_sha256'] for r in rows})))
 report=load(p/f'image_heads_step_{step:07d}/report.json')
 resources.append(dict(step=step,cases=len(report['records']),**{k:report[k] for k in ['checkpoint_load_seconds','total_generation_seconds','peak_allocated_vram_gib']}))
x.update(export_source_sha256=sha(Path(__file__)),qualification_verdict='fails current language criteria',result_date='2026-11-11',result='The rank-repaired writer produces no credited readable complete output or exact QA answer on this registered development cohort. Removing the measured rank restriction was not sufficient to solve the language failure.',review_summary={**frozen,'frozen_receipt_sha256':sha(b/'readings_frozen.json'),'readings_file':'compact_visual_span_full_rank_readings_20261111.json'},normal_output_counts=normal,generation_resources=resources,trained_cold_replay_evidence='compact_visual_span_full_rank_cold_20261111.json',trained_cold_replay_evidence_sha256=sha(e/'compact_visual_span_full_rank_cold_20261111.json'),training=dict(updates=fit['updates'],training_sequences=fit['training_records'],target_glyph_positions=2576239,causal_state_targets=fit['supervised_states_per_epoch'],peak_allocated_vram_gib=fit['completed_shard_peak_allocated_vram_gib'],elapsed_seconds=fit['elapsed_seconds'],resource_wait_seconds=fit['resource_wait_seconds']),raw_examples_directory=out.name,next_decision='Preserve both failed fits. Localize visual access, conditional writing under reference history, sampler discretization and generated-history drift before choosing another training change. No new architecture or longer fit is selected by these results.',goal_achieved=False,trained_visual_language_result_means_evaluation_not_success=True,control_evidence='compact_visual_span_learning_20261111.json',control_evidence_sha256=sha(e/'compact_visual_span_learning_20261111.json'),final_complete_exact_qa_difference_from_control=0,final_both_correct_pairs_difference_from_control=0)
serialized=json.dumps(x,ensure_ascii=False,indent=2)+'\n'
assert '/home/' not in serialized
(R/e/'compact_visual_span_full_rank_learning_20261111.json').write_text(serialized)
print(json.dumps({'normal':normal,'resources':resources,'summary_sha256':sha(a/'summary.json'),'export_sha256':sha(e/'compact_visual_span_full_rank_learning_20261111.json'),'training':x['training']},indent=2))
