#!/usr/bin/env python3
"""Bounded CPU export of completed evidence; preserve the current fit owner."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import build_ranking_completed_article as base
from prepare_visual_context_evidence import ROOT,HERE,protected_sources
from prepare_glyph_score_gap_figure import owners

SKILL=Path('/home/lachlan/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations')
RUNTIME=Path('/home/lachlan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
    with (ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);protected=owners();protected_sources()
        h=base.host();assert h['available_gib']>=48 and h['some']<=.5 and h['full']<=.1
        stage=ROOT/f'artifacts/visual_context_evidence_publication_v1/figure_attempt{a.attempt:02}';stage.mkdir(exist_ok=False)
        base.STAGE=stage;(stage/'node_modules').symlink_to(RUNTIME/'node_modules',target_is_directory=True)
        source=HERE/'visual_context_evidence_figures.mjs';target=stage/source.name;shutil.copy2(source,target)
        costs=dict(child_seconds=0.,minimum_available_host_gib=h['available_gib'],peak_child_tree_rss_gib=0.,children=[])
        prior=sorted(p for p in stage.parent.glob('figure_attempt*/child_costs.json') if p.parent!=stage)
        if prior:costs=base.load(prior[-1])
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
            RUNTIME_NODE_MODULES=str(RUNTIME/'node_modules'),RUNTIME_NODE=str(RUNTIME/'bin/node'),
            RUNTIME_PYTHON=os.sys.executable,RUNTIME_BIN_DIR=str(RUNTIME/'bin'))
        base.run([str(RUNTIME/'bin/node'),str(target),str(ROOT),str(stage),str(SKILL),os.sys.executable],stage/'export.log',env,costs)
        protected_sources();assert owners()==protected
        costs.update(existing_shared_runtime_reused=True,dependency_loader_tool_unavailable=True,new_model_calls=0)
        (stage/'costs.json').write_text(json.dumps(costs,indent=2)+'\n')
        print(json.dumps({k:v for k,v in costs.items() if k!='children'}))
