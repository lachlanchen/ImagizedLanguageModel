#!/usr/bin/env python3
"""Check the exact admission-only delta from preserved numerical sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD_INFER=ROOT/'scripts/infer_trainable_visual_context.py'
NEW_INFER=ROOT/'scripts/infer_visual_context_deployment.py'
OLD_GUARD=ROOT/'ilm/visual_lm/glyph_runtime_resources.py'
BRIDGE=ROOT/'scripts/train_semantic_glyph_bridge.py'
NEW_GUARD=ROOT/'ilm/visual_lm/visual_context_deployment_resources.py'
OLD_IMPORT='from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu'
NEW_IMPORT='from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu'
DISK_CHECK="        require(shutil.disk_usage(ROOT).free >= 48*1024**3, 'deployment disk reserve unavailable')\n"
HEADER='''"""Deployment admission without obsolete-job JSON dependencies."""
from contextlib import contextmanager
import fcntl
import os
import shutil
import subprocess

from ilm.image_edit.baseline import ROOT
from scripts.train_semantic_glyph_bridge import project_processes
from scripts.score_pixel_language_first_blocks import host_resources, require


'''


def function_source(path,name):
    source=path.read_text()
    node,=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(node.decorator_list)!=1 or not isinstance(node.decorator_list[0],ast.Name) or node.decorator_list[0].id!='contextmanager':
        raise ValueError('Preserve context-manager structure')
    return ast.get_source_segment(source,node)+'\n'


def expected_sources():
    original=OLD_INFER.read_text()
    if original.count(OLD_IMPORT)!=1:raise ValueError('Original admission import differs')
    inference=original.replace(OLD_IMPORT,NEW_IMPORT)
    bridge=function_source(BRIDGE,'bridge_gpu')
    dependency="    owner_gone(ROOT/'artifacts/pixel_language_diversity_training_v2')\n"
    if bridge.count(dependency)!=1 or bridge.count('        import torch\n')!=1:
        raise ValueError('Original bridge structure differs')
    bridge=bridge.replace('def bridge_gpu(prepared):','def deployment_bridge_gpu(prepared):',1)
    bridge=bridge.replace(dependency,'').replace('        import torch\n',DISK_CHECK+'        import torch\n',1)
    wrapper=function_source(OLD_GUARD,'full_reader_gpu').replace('with bridge_gpu(', 'with deployment_bridge_gpu(',1)
    guard=HEADER+'@contextmanager\n'+bridge+'\n\n@contextmanager\n'+wrapper
    return inference,guard


def verify_text(inference,guard):
    expected_inference,expected_guard=expected_sources()
    if inference!=expected_inference:raise ValueError('Numerical inference changed beyond the admission import')
    if guard!=expected_guard:raise ValueError('Resource guard changed beyond removing obsolete ownership JSON and adding a disk floor')


def verify_sources():
    verify_text(NEW_INFER.read_text(),NEW_GUARD.read_text())
    paths=[OLD_INFER,NEW_INFER,OLD_GUARD,BRIDGE,NEW_GUARD,Path(__file__).resolve()]
    return dict(numerical_inference_source_identical_except_admission_import=True,
        bridge_identical_except_obsolete_owner_check_removed_and_disk_floor_added=True,
        wrapper_and_resource_thresholds_preserved=True,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        language_quality_proved=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:
        for path,source in zip((NEW_INFER,NEW_GUARD),expected_sources()):
            with path.open('x') as f:f.write(source)
    print(json.dumps(verify_sources()))
