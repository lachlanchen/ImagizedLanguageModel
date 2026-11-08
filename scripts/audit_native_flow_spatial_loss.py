#!/usr/bin/env python3
"""Measure spatial loss/gradient shares in one real saved denoising forward.

No model is loaded. Autograd acts on the saved prediction tensor only, not
on a neural network. Image-space alignment is descriptive, not causal.
"""
import argparse
import json
import math
import os
from pathlib import Path

from ilm.image_edit.baseline import ROOT,save_json,sha


def audit(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('hide CUDA for the saved-tensor audit')
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F
    from ilm.image_edit.visual_flow_learning import velocity_loss
    torch.set_num_threads(4)
    source=ROOT/'artifacts/visual_flow_step_preflight_v2/execution'
    executed=json.loads((source/'report.json').read_text());learning=executed['learning']
    prepared=json.loads((source.parent/'preparation.json').read_text());row=prepared['selected_cases'][0]
    data=ROOT/'artifacts/visual_flow_pages_v1';target=data/row['target']
    if sha(target)!=row['target_sha256'] or row['target_sha256']!=learning['target_sha256']:
        raise ValueError('owning actual training target changed')
    if row['input_sha256']!=learning['input_sha256'] or learning['actual_optimizer_updates']!=1:
        raise ValueError('actual single-step ownership differs')
    path=source/'actual_training_tensors.pt';actual=torch.load(path,map_location='cpu',weights_only=True)
    prediction=actual['predicted_velocity'].detach().requires_grad_()
    loss=velocity_loss(prediction,actual['target_velocity']);loss.backward()
    if not math.isclose(float(loss.detach()),learning['loss'],rel_tol=1e-6,abs_tol=1e-8):
        raise ValueError('actual native training loss does not reproduce')
    error=(prediction.detach().float()-actual['target_velocity'].float()).square().mean((0,1))
    gradient=prediction.grad.float().square().mean((0,1))
    image=np.array(Image.open(target).convert('L'))<128
    h,w=error.shape
    if image.shape!=(1024,1024) or h!=128 or w!=128:raise ValueError('declared spatial geometry changed')
    ink=torch.from_numpy(image).float()[None,None]
    aligned=F.max_pool2d(ink,8)[0,0].bool()

    def region(name,mask):
        return dict(region=name,latent_cells=int(mask.sum()),area_fraction=float(mask.float().mean()),
            squared_error_fraction=float(error[mask].sum()/error.sum()),
            squared_prediction_gradient_fraction=float(gradient[mask].sum()/gradient.sum()),
            mean_error_inside=float(error[mask].mean()),mean_error_outside=float(error[~mask].mean()),
            mean_error_inside_to_outside=float(error[mask].mean()/error[~mask].mean()))

    regions=[]
    for radius in (0,1,3):
        mask=aligned if radius==0 else F.max_pool2d(aligned.float()[None,None],2*radius+1,1,radius)[0,0].bool()
        regions.append(region('ink_aligned_halo_'+str(radius),mask))
    strip=torch.zeros_like(aligned);strip[:32,:64]=True
    regions.append(region('fixed_512x256_answer_region',strip))
    output.mkdir(parents=True,exist_ok=False)
    result=dict(implementation_sha256=sha(Path(__file__)),native_loss_implementation_sha256=sha(ROOT/'ilm/image_edit/visual_flow_learning.py'),
        actual_training_tensors_sha256=sha(path),actual_step_report_sha256=sha(source/'report.json'),
        target_sha256=sha(target),source_case_id=row['id'],source_target_black_pixels=int(image.sum()),
        actual_native_loss=float(loss.detach()),actual_sigma=float(actual['sigmas'].item()),
        prediction_dtype=str(prediction.dtype),prediction_shape=list(prediction.shape),regions=regions,
        spatial_alignment='any target ink in each8x8 image region; optional latent-grid max-pool halo',
        gradient_scope='saved prediction tensor; network parameter-gradient attribution not measured',
        one_pre_adaptation_training_draw_only=True,model_loaded=False,new_neural_forwards=0,optimizer_updates=0,
        actual_autograd_on_saved_prediction=True,loss_or_training_recipe_changed=False,
        causal_bottleneck_identified=False,language_learning_validated=False,general_ilm_goal_complete=False)
    if torch.cuda.is_initialized():raise ValueError('CPU audit initialized CUDA')
    save_json(output/'report.json',result)
    print(json.dumps(dict(loss=result['actual_native_loss'],sigma=result['actual_sigma'],regions=regions),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args().output.resolve())
