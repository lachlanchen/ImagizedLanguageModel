"""Lightweight source/pixel audit for the completed motor-ambiguity experiments."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EVIDENCE=ROOT/'references/evidence'
INDICES=(1,1362,2724,4086)


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())


def collect(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for raw-data collection')
    require(not output.exists(),'preserve every prior source audit')
    bound={}
    def read(relative):
        p=EVIDENCE/relative;bound[str(p.relative_to(ROOT))]=sha(p);return load(p)
    geometry=read('semantic_glyph_geometry/report.json')
    require(geometry['summary']==dict(normal_cases=16,normal_top_reference_form=9,
        normal_reference_in_top3=11,normal_first_image_exact=1),'first-state data differs')
    spectrum=read('glyph_readout_rank_v1/report.json')
    best=next(r for r in spectrum['optimal_affine_rank_bounds'] if r['affine_rank']==2048)
    require(spectrum['neural_forward_calls']==0 and not spectrum['CUDA_initialized'],'spectrum is not a CPU-only diagnosis')
    head=read('semantic_readout_distillation_v1/qualify/report.json')
    contrast=read('contrastive_glyph_readout_v1/qualify/report.json')
    require(not head['compiler_gate_passed'] and not contrast['compiler_gate_passed'],'compiler gate interpretation differs')
    tests=[]
    for folder,name,parameters in [('robust_glyph_decoder_v1','Noise-trained D',10792449),
        ('semantic_motor_flow_v1','Dense motor flow',168366080),('spatial_semantic_motor_v1','Spatial image flow',3892737)]:
        r=read(folder+'/qualify/report.json');fit=read(folder+'/fit/report.json')
        require(not r['primitive_gate_passed'],'candidate unexpectedly qualified')
        count=fit.get('trained_parameters',fit.get('flow_parameters'))
        require(count==parameters,'trainable parameter count differs')
        s=r['summary']
        if folder=='robust_glyph_decoder_v1':
            clean=s['robust/clean'];mixed=s['robust/mixture045'];c=clean['mean_f1'];m=mixed['mean_f1']
        else:
            clean=s['clean'];mixed=s['mixture045'];c=clean['mean_selected_f1'];m=mixed['mean_selected_f1']
        require(clean['forms']==mixed['forms']==4087,'known-form denominator differs')
        tests.append(dict(folder=folder,name=name,trained_parameters=parameters,forms=4087,
            clean_f1=c,mixed_f1=m,primitive_gate_passed=False,fit_seconds=fit['elapsed_seconds'],
            peak_allocated_vram_gib=fit['peak_allocated_vram_gib']))
    source_names={
        'D':('robust_glyph_decoder_v1/baseline_mixture045_all.png',
             'robust_glyph_decoder_v1/visual_review.json','all_outputs_lossless'),
        'Noise D':('robust_glyph_decoder_v1/robust_mixture045_all.png',
             'robust_glyph_decoder_v1/visual_review.json','all_outputs_lossless'),
        'Dense flow':('semantic_motor_flow_v1/mixture045_all_outputs_ink.png',
             'semantic_motor_flow_v1/retention.json','all_actual_selected_and_candidate_pixels_retained'),
        'Spatial flow':('spatial_semantic_motor_v1/mixture045_all_outputs_ink.png',
             'spatial_semantic_motor_v1/visual_review.json','all_actual_pixels_retained')}
    images={};sources={}
    for name,(relative,manifest,key) in source_names.items():
        path=EVIDENCE/relative;receipt=read(manifest)
        require(sha(path)==receipt[key][path.name]['png_sha256'],'raw packed source differs')
        bound[str(path.relative_to(ROOT))]=sha(path);images[name]=Image.open(path).convert('L');sources[name]=path
    reference=np.asarray(images['D'])
    for relative,manifest,key in [
        ('semantic_motor_flow_v1/mixture045_all_outputs_baseline_ink.png','semantic_motor_flow_v1/retention.json','all_actual_selected_and_candidate_pixels_retained'),
        ('spatial_semantic_motor_v1/mixture045_all_outputs_baseline_ink.png','spatial_semantic_motor_v1/visual_review.json','all_actual_pixels_retained')]:
        p=EVIDENCE/relative;r=read(manifest)
        require(sha(p)==r[key][p.name]['png_sha256'],'baseline packed pixels changed')
        require(np.array_equal(reference,np.asarray(Image.open(p).convert('L'))),'original decoder baselines are not pixel-identical')
        bound[str(p.relative_to(ROOT))]=sha(p)
    require(all(im.size==(2048,2048) for im in images.values()),'full packed grid geometry differs')
    import torch
    torch.set_num_threads(2)
    data_path=ROOT/'artifacts/semantic_glyph_bridge_v1/training.pt'
    primitive=read('semantic_glyph_bridge/preparation.json')
    require(sha(data_path)==primitive['training_sha256'],'canonical source tensor changed')
    data=torch.load(data_path,weights_only=True,map_location='cpu')
    require(data['ink'].shape==(4087,1,32,32),'canonical glyph source shape differs')
    output.mkdir(parents=True)
    cells=[]
    for i,index in enumerate(INDICES):
        canonical=255*(1-data['ink'][index,0].numpy().astype(np.uint8))
        require(bool((canonical==0).any()),'selected display index is empty')
        path=output/f'row{i}_target.png';Image.fromarray(canonical).save(path)
        cells.append(dict(row=i,index=index,column='Target',file=path.name,sha256=sha(path),
            source='artifacts/semantic_glyph_bridge_v1/training.pt',source_sha256=sha(data_path),crop=None))
        box=((index%64)*32,(index//64)*32,(index%64+1)*32,(index//64+1)*32)
        for name,im in images.items():
            path=output/f'row{i}_{name.lower().replace(" ","_")}.png';im.crop(box).save(path)
            require(np.array_equal(np.asarray(Image.open(path)),np.asarray(im)[box[1]:box[3],box[0]:box[2]]),'cell pixel extraction changed values')
            cells.append(dict(row=i,index=index,column=name,file=path.name,sha256=sha(path),
                source=str(sources[name].relative_to(ROOT)),source_sha256=sha(sources[name]),crop=list(box)))
    result=dict(first_state=geometry['summary'],regression=dict(coordinates=4096,affine_rank=2048,
        best_span_mse=best['minimum_uniform_coordinate_MSE'],trained_span_mse=spectrum['trained_output_span']['mean_uniform_coordinate_MSE']),
        compilers=dict(original=head['summary'],contrastive=contrast['summary']),motor_tests=tests,
        original_decoder_mixed_baselines_pixel_identical=True,indices=list(INDICES),cells=cells,
        source_files_sha256=bound,canonical_training_sha256=sha(data_path),
        collector_sha256=sha(__file__),neural_forward_calls=0,images_repaired=False,
        cells_are_complete_32x32_glyphs=True,genuine_curriculum_results_included=False)
    (output/'sources.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    r=collect(p.parse_args().output.resolve())
    print(json.dumps(dict(cells=len(r['cells']),indices=r['indices'],baseline_pixels_identical=True,neural_calls=0)))
