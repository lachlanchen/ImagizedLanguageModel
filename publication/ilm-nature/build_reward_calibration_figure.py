#!/usr/bin/env python3
"""Bounded native evidence export; no GPU, model or office process."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import shutil
import build_ranking_completed_article as base
from prepare_glyph_score_gap_figure import owners

ROOT, HERE = base.ROOT, base.HERE
STAGE = ROOT / 'artifacts/reward_calibration_publication_v1/figure_attempt01'
SKILL = Path('/home/lachlan/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations')
RUNTIME = Path('/home/lachlan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args(); assert args.attempt >= 1
    STAGE = STAGE.parent / f'figure_attempt{args.attempt:02}'
    with (ROOT / 'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protected = owners(); base.model_sources()
        assert base.host()['available_gib'] >= 48 and base.host()['some'] <= .5 and base.host()['full'] <= .1
        STAGE.mkdir(exist_ok=False); base.STAGE = STAGE
        (STAGE / 'node_modules').symlink_to(RUNTIME / 'node_modules', target_is_directory=True)
        source = HERE / 'reward_calibration_figures.mjs'; target = STAGE / source.name; shutil.copy2(source, target)
        costs = dict(child_seconds=0., minimum_available_host_gib=base.host()['available_gib'], peak_child_tree_rss_gib=0., children=[])
        previous = sorted(p for p in STAGE.parent.glob('figure_attempt*/child_costs.json') if p.parent != STAGE)
        if previous: costs = base.load(previous[-1])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   RUNTIME_NODE_MODULES=str(RUNTIME / 'node_modules'), RUNTIME_NODE=str(RUNTIME / 'bin/node'),
                   RUNTIME_PYTHON=os.sys.executable, RUNTIME_BIN_DIR=str(RUNTIME / 'bin'))
        base.run([str(RUNTIME / 'bin/node'), str(target), str(ROOT), str(STAGE), str(SKILL), os.sys.executable],
                 STAGE / 'export.log', env, costs)
        base.model_sources(); assert owners() == protected
        costs.update(existing_shared_runtime_reused=True, dependency_loader_tool_unavailable=True,
                     protected_model_owners_preserved=True, new_model_calls=0)
        (STAGE / 'costs.json').write_text(json.dumps(costs, indent=2)+'\n')
        print(json.dumps({k:v for k,v in costs.items() if k != 'children'}))
