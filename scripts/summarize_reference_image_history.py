#!/usr/bin/env python3
"""CPU-only independent recount and complete pixel-board export of supplied histories."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from scripts.diagnose_reference_image_history import RUN, validate, verify_case, load
from scripts.score_pixel_language_first_blocks import require

PUBLIC = ROOT/'references/evidence/reference_image_history_20261112'
LEARNING = ROOT/'references/evidence/reference_image_history_learning_20261112.json'


def recount(actual, gold):
    require(actual.shape == gold.shape and set(np.unique(actual)).issubset({0,1}) and
            set(np.unique(gold)).issubset({0,1}), 'Binary equal-shaped arrays required')
    rows = []
    for a, b in zip(actual.reshape(len(actual),-1), gold.reshape(len(gold),-1)):
        # Count independently from the runtime metric implementation.
        intersection = int(np.count_nonzero(a*b)); total_a = int(np.count_nonzero(a))
        total_b = int(np.count_nonzero(b)); fp = total_a-intersection; fn = total_b-intersection
        denominator = total_a+total_b
        rows.append(dict(tp=intersection, fp=fp, fn=fn, target_ink=total_b, predicted_ink=total_a,
            pixel_exact=bool(np.count_nonzero(a != b) == 0),
            foreground_f1=2*intersection/denominator if denominator else 1.))
    return rows


def aggregate(rows):
    scores = [x['foreground_f1'] for x in rows]
    return dict(positions=len(rows), exact=sum(x['pixel_exact'] for x in rows),
        mean_f1=float(np.mean(scores)) if scores else None,
        median_f1=float(np.median(scores)) if scores else None,
        at_least_095=sum(x>=.95 for x in scores))


def boards(public, cases):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    manifests = []
    for start in range(0,len(cases),4):
        group = cases[start:start+4]
        heights = [56+114*math.ceil(max(len(c['gold_ink']),len(c['autonomous_ink']))/32) for c in group]
        canvas = Image.new('L',(1152,40+sum(heights)),240); draw = ImageDraw.Draw(canvas)
        draw.text((8,8),'Supplied-history diagnostic; stopping bypassed. Grey cells are absent, not blank predictions.',font=font,fill=0)
        cells = []; top = 40
        for case, height in zip(group,heights):
            draw.text((8,top),case['id']+' | reference length='+str(len(case['gold_ink']))+
                ' | correct-history premature stops='+str(case['premature_stops'])+
                ' | terminal stop='+str(case['terminal_stop']),font=font,fill=0)
            for row in range(math.ceil(max(len(case['gold_ink']),len(case['autonomous_ink']))/32)):
                for track,(key,label) in enumerate([('gold_ink','Reference'),('forced_ink','Correct prior'),('autonomous_ink','Autonomous')]):
                    y = top+30+114*row+32*track; draw.text((4,y+7),label,font=font,fill=0)
                    for position in range(32*row,min(32*(row+1),len(case[key]))):
                        x = 112+32*(position%32)
                        canvas.paste(Image.fromarray(255-255*case[key][position,0]),(x,y))
                        cells.append(dict(case=case['id'],array=key,position=position,x=x,y=y))
            top += height
        path = public/f'board_{start//4+1:02}.png'; canvas.save(path)
        actual = np.asarray(Image.open(path).convert('L'))
        indexed = {c['id']:c for c in group}
        for cell in cells:
            expected = 255-255*indexed[cell['case']][cell['array']][cell['position'],0]
            require(np.array_equal(actual[cell['y']:cell['y']+32,cell['x']:cell['x']+32],expected),
                    'Displayed original pixel differs')
        manifests.append(dict(file=path.name,sha256=sha(path),cells=cells))
    save_json(public/'display_manifest.json',dict(boards=manifests,all_displayed_pixels_verified=True,
        actual_visual_review_complete=False,reference_conditioned=True,
        terminal_predictions_not_shown_as_target_glyphs=True))
    return len(manifests), sum(len(x['cells']) for x in manifests)


def summarize(run, public):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only recount')
    require(not public.exists() and not LEARNING.exists(), 'Preserve completed evidence')
    torch.set_num_threads(2); r = validate(run); execution = run/'execution'
    report = load(execution/'report.json'); registration_sha = sha(run/'registration.json')
    require(report['all64_complete'] and report['registration_sha256'] == registration_sha, 'Complete fixed execution required')
    prepared = torch.load(run/'prepared.pt', weights_only=True, map_location='cpu')
    public.mkdir(parents=True); records = []; display = []; metrics = {}; stops = {}
    for job in r['jobs']:
        folder = execution/job['id']; record = verify_case(folder,job,registration_sha)
        trace = torch.load(folder/'trace.pt', weights_only=True, map_location='cpu')
        raw = dict(np.load(folder/'pixels.npz')); n = job['reference_glyphs']
        expected = prepared['gold_ink'][job['reference_offset']:job['reference_offset']+n].numpy()
        require(np.array_equal(raw['gold_ink'],expected) and np.array_equal(raw['forced_ink'],trace['predicted_ink'].numpy()),
                'Raw images differ from bound targets or trace')
        require(np.array_equal(raw['forced_stops'],trace['stop_probabilities']) and len(raw['forced_stops']) == n+1,
                'Stop boundaries differ')
        require(torch.equal(trace['states'][0],prepared['initial_states'][job['id']]), 'Initial state differs')
        original = torch.load(ROOT/job['original_output'], weights_only=True, map_location='cpu')
        require(np.array_equal(raw['autonomous_ink'],((1-original['images'])/2).to(torch.uint8).numpy()) and
                np.array_equal(raw['autonomous_stops'],original['stop_probabilities']), 'Original autonomous trace differs')
        require(len(trace['reference_feedback']) == n, 'Missing reference image feedback')
        for i, feedback in enumerate(trace['reference_feedback']):
            require(tensor_hash(1-2*torch.from_numpy(expected[i:i+1]).float()) == feedback['image_sha256'],
                    'Reference-feedback pixels differ')
        forced = recount(raw['forced_ink'][:-1],expected)
        require(forced == record['per_position_canonical_pixels'], 'Independent per-position recount differs')
        premature = np.flatnonzero(raw['forced_stops'][:-1]>=.5).tolist()
        terminal = bool(raw['forced_stops'][-1]>=.5)
        require(premature == record['premature_stop_positions'] and terminal == record['terminal_stop'], 'Independent stop recount differs')
        shared = min(n,len(raw['autonomous_ink'])); autonomous = recount(raw['autonomous_ink'][:shared],expected[:shared]) if shared else []
        overlap = min(len(trace['states']),len(original['states']))
        a = trace['states'][:overlap].numpy().astype(np.float64); b = original['states'][:overlap].numpy().astype(np.float64)
        similarity = (a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
        distance = np.linalg.norm(a-b,axis=1)/np.maximum(np.linalg.norm(b,axis=1),1e-12)
        require(distance[0] == 0, 'Initial hidden state not exactly equal')
        for pool,values in [('first',forced[:1]),('later',forced[1:]),('all',forced),('autonomous_shared',autonomous),('forced_shared',forced[:shared])]:
            metrics.setdefault(job['layout'],{}).setdefault(pool,[]).extend(values)
        s = stops.setdefault(job['layout'],dict(nonterminal=0,false_stops=0,terminal=0,terminal_correct=0,cases_with_early_stop=0))
        s['nonterminal']+=n;s['false_stops']+=len(premature);s['terminal']+=1;s['terminal_correct']+=terminal;s['cases_with_early_stop']+=bool(premature)
        path = public/(job['id']+'.npz');shutil.copy2(folder/'pixels.npz',path)
        records.append(dict(job=job,pixels_file=path.name,pixels_sha256=sha(path),trace_sha256=record['trace_sha256'],
            canonical_forced_positions=forced,canonical_autonomous_shared_positions=autonomous,
            missing_autonomous_target_positions=n-shared,extra_autonomous_positions=max(0,len(raw['autonomous_ink'])-n),
            premature_stop_positions=premature,terminal_stop=terminal,
            same_index_state_cosine=similarity.tolist(),same_index_relative_state_distance=distance.tolist(),
            semantic_alignment_of_autonomous_positions=False))
        display.append(dict(id=job['id'],gold_ink=expected,forced_ink=raw['forced_ink'][:-1],autonomous_ink=raw['autonomous_ink'],
                            premature_stops=len(premature),terminal_stop=terminal))
    grouped = {layout:{pool:aggregate(rows) for pool,rows in groups.items()} for layout,groups in metrics.items()}
    for layout,s in stops.items():
        s['false_stop_fraction']=s['false_stops']/s['nonterminal'];s['terminal_recognition_fraction']=s['terminal_correct']/s['terminal']
        s['stop_diagnostic_gate_passed']=s['false_stop_fraction']<=.05 and s['terminal_recognition_fraction']>=.8
    gates = {layout:values['later']['median_f1']>=.99 and values['later']['at_least_095']/values['later']['positions']>=.9
             for layout,values in grouped.items()}
    require(len(records)==64 and sum(x['all']['positions'] for x in grouped.values())==1490 and sum(x['terminal'] for x in stops.values())==64,
            'Coverage denominator differs')
    nboards, tiles = boards(public,display)
    value = dict(schema='reference-image-history-learning-v1',registration_sha256=registration_sha,
        execution_report_sha256=sha(execution/'report.json'),sources_sha256=r['source_sha256'],records=records,
        groups=grouped,stopping=stops,later_physical_gates=gates,all64_initial_states_bit_exact=True,
        all64_autonomous_pixel_and_stop_replays_exact=True,independent_pixel_and_stop_recount_passed=True,
        display_boards=nboards,displayed_original_tiles=tiles,all_displayed_pixels_independently_checked=True,
        visual_review_complete=False,reference_conditioned=True,supplied_target_length=True,
        new_optimizer_updates=0,new_autonomous_generations=0,new_native_helper_calls=0,
        original_complete_language_scores_unchanged=True,reused_development_cohort=True,
        independent_human_adjudication=False,canonical_mismatch_not_semantic_error=True,
        same_index_comparison_not_semantic_alignment=True,goal_achieved=False,
        resources={k:report[k] for k in ['elapsed_seconds','resource_wait_seconds','peak_allocated_vram_gib','minimum_available_gib','loading_seconds']})
    save_json(LEARNING,value)
    save_json(public/'export.json',dict(learning_sha256=sha(LEARNING),files_sha256={p.name:sha(p) for p in sorted(public.iterdir()) if p.is_file()},
                                      actual_visual_review_complete=False))
    print(json.dumps(dict(groups=grouped,stopping=stops,gates=gates,boards=nboards,tiles=tiles),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--public',type=Path,default=PUBLIC);a=p.parse_args();summarize(a.run,a.public)
