#!/usr/bin/env python3
"""CPU-only spectral limits of the existing affine semantic output span."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.score_pixel_language_first_blocks import host_resources,require


def main(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only spectral analysis requires CUDA hidden')
    own='scripts/diagnose_glyph_readout_rank.py'
    require(subprocess.check_output(['git','show','HEAD:'+own],cwd=ROOT)==(ROOT/own).read_bytes(),'commit spectral source first')
    require(not output.exists(),'preserve every previous rank observation')
    import torch
    torch.set_num_threads(4);began=time.perf_counter();before=host_resources()
    require(before['available_gib']>=24. and before['some_pressure_avg10']<=.5 and before['full_pressure_avg10']<=.1,'CPU analysis resource reserve unavailable')
    primitive=ROOT/'artifacts/semantic_glyph_bridge_v1';language=ROOT/'artifacts/semantic_glyph_language_v1'
    pmeta=json.loads((primitive/'preparation.json').read_text());report=json.loads((language/'fit/report.json').read_text())
    data_path=primitive/'training.pt';state_path=language/'fit/checkpoint_latest.pt'
    require(sha(data_path)==pmeta['training_sha256'] and sha(state_path)==report['checkpoint_sha256'],'completed anchor/model state changed')
    data=torch.load(data_path,weights_only=True,map_location='cpu')
    z=((data['semantic']-data['mean'])/data['scale']).double()
    mean=z.mean(0);centered=z-mean
    covariance=centered.T@centered/len(z)
    values=torch.linalg.eigvalsh(covariance).clamp_min(0)
    total=float(values.sum());dimension=z.shape[1]
    ranks=[512,1024,2048,3072,4086,4096];bounds=[]
    for rank in ranks:
        tail=float(values[:max(0,dimension-rank)].sum())
        bounds.append(dict(affine_rank=rank,minimum_uniform_coordinate_MSE=tail/dimension,
            maximum_explained_variance_fraction=1-tail/total))
    saved=torch.load(state_path,weights_only=True,map_location='cpu')['model']
    w=saved['readout.3.weight'].double();bias=saved['readout.3.bias'].double()
    require(w.shape==(4096,2048),'original readout architecture differs')
    q,r=torch.linalg.qr(w,mode='reduced')
    shifted=z-bias
    residual=shifted.square().sum(-1)-(shifted@q).square().sum(-1)
    # A QR basis bounds distance to this trained output span. It does not run
    # G or D and does not infer any physical-image or language accuracy.
    require(float(residual.min())>=-1e-6,'unstable span distance')
    after=host_resources();require(after['available_gib']>=24.,'CPU spectral exit reserve reached')
    output.mkdir(parents=True)
    save_json(output/'report.json',dict(source_sha256=sha(ROOT/own),primitive_training_sha256=sha(data_path),
        checkpoint_sha256=sha(state_path),forms=len(z),coordinates=dimension,
        centered_covariance_total_variance=total,
        covariance_effective_rank=total**2/float(values.square().sum()),
        optimal_affine_rank_bounds=bounds,
        trained_output_span=dict(columns=w.shape[1],minimum_absolute_QR_diagonal=float(r.diagonal().abs().min()),
            mean_uniform_coordinate_MSE=float(residual.clamp_min(0).mean()/dimension),
            minimum_form_MSE=float(residual.clamp_min(0).min()/dimension),
            maximum_form_MSE=float(residual.clamp_min(0).max()/dimension)),
        eigenvalues=values.tolist(),elapsed_seconds=time.perf_counter()-began,
        resources_before=before,resources_after=after,CUDA_initialized=torch.cuda.is_initialized(),
        neural_forward_calls=0,optimizer_updates=0,images_generated=0,
        pixel_or_language_failure_not_proved_by_latent_rank_bound=True))
    print(json.dumps(dict(bounds=bounds,current_span_MSE=float(residual.mean()/dimension),seconds=time.perf_counter()-began)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
