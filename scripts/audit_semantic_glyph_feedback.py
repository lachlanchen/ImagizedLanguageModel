#!/usr/bin/env python3
"""Independently reread every actual emitted glyph; no native language body."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.score_pixel_language_first_blocks import require,host_resources
from scripts.train_semantic_glyph_bridge import bridge_gpu


def load(path):return json.loads(Path(path).read_text())


def run(output):
    require(not output.exists(),'preserve prior feedback audits')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','deterministic cuBLAS required')
    import torch
    from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge,tensor_hash
    jobs=[]
    for run,stage,key in [('semantic_glyph_language_v1','assessment','file'),
                          ('semantic_glyph_counterfactual_v1','execution','tensor')]:
        folder=ROOT/'artifacts'/run/stage
        require(load(folder/'status.json')['phase']=='completed','autonomous generation must finish')
        for row in load(folder/'report.json')['records']:
            path=folder/row[key];require(sha(path)==row['sha256'],'original stream changed')
            jobs.append(dict(path=str(path.relative_to(ROOT)),sha256=row['sha256']))
    checkpoint=ROOT/'artifacts/semantic_glyph_language_v1/fit/checkpoint_latest.pt'
    require(sha(checkpoint)==load(checkpoint.parent/'report.json')['checkpoint_sha256'],'learned bridge changed')
    code={name:sha(ROOT/name) for name in ['scripts/audit_semantic_glyph_feedback.py','ilm/visual_lm/semantic_glyph_loop.py']}
    for name,digest in code.items():
        import hashlib
        require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)).hexdigest()==digest,
                'commit feedback verifier first')
    output.mkdir(parents=True);began=time.perf_counter();count=0;maximum_error=0.;records=[]
    try:
        with bridge_gpu(dict(minimum_entry_ram_gib=64.)) as (device,admission),torch.no_grad():
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
            bridge=SemanticGlyphBridge().to(device)
            bridge.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu')['model'])
            bridge.requires_grad_(False).eval();cpu=copy.deepcopy(bridge).cpu()
            torch.set_num_threads(2)
            for job in jobs:
                value=torch.load(ROOT/job['path'],weights_only=True,map_location='cpu')
                for i,feedback in enumerate(value['feedback']):
                    image=value['images'][i:i+1]
                    require(tensor_hash(image)==feedback['image_sha256'],'actual-image hash differs')
                    embedded=bridge.feedback_embedding(image.to(device))
                    require(tensor_hash(embedded)==feedback['embedding_sha256'],'cold GPU feedback embedding is not bit-exact')
                    error=float((cpu.feedback_embedding(image)-embedded.cpu()).abs().max())
                    maximum_error=max(error,maximum_error);require(error<1e-4,'CPU feedback tolerance exceeded')
                    count+=1
                require(torch.isfinite(value['states']).all(),'nonfinite generated contextual state')
                records.append(dict(**job,emitted_glyphs=len(value['images']),
                    all_actual_images_and_GPU_feedback_embeddings_bit_exact=True))
                resources=host_resources()
                require(resources['available_gib']>=32 and resources['some_pressure_avg10']<=.5
                        and resources['full_pressure_avg10']<=.1,'feedback audit resource reserve')
                save_json(output/'status.json',dict(phase='verifying',streams=len(records),glyphs=count,pid=os.getpid()))
            admission.pop('project_processes',None)
            save_json(output/'report.json',dict(streams=len(jobs),glyphs=count,records=records,
                maximum_CPU_GPU_feedback_error=maximum_error,CPU_tolerance=1e-4,
                all_cold_GPU_embedding_hashes_bit_exact=True,
                checkpoint_sha256=sha(checkpoint),code_sha256=code,admission=admission,
                elapsed_seconds=time.perf_counter()-began,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                native_context_body_loaded=False,teacher_loaded=False,optimizer_updates=0,
                recording_and_feedback_check_is_not_semantic_validation=True))
            save_json(output/'status.json',dict(phase='completed',pid=None,streams=len(jobs),glyphs=count))
    except BaseException as error:
        save_json(output/'status.json',dict(phase='failed',reason=repr(error),streams=len(records),glyphs=count,pid=None));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
