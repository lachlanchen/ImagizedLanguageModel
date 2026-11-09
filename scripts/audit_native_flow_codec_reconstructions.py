#!/usr/bin/env python3
"""Audit the eight already-saved native codec reconstructions on CPU."""
import argparse
import json
import os
from pathlib import Path
import shutil

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.image_edit.native_flow_assessment import authenticate_native_png
from scripts.evaluate_image_edit_baseline import engine, normalize, recognize


def audit(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('hide CUDA for saved-image and offline OCR auditing')
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(4)
    preflight = ROOT/'artifacts/visual_flow_step_preflight_v2'
    data = ROOT/'artifacts/visual_flow_pages_v1'
    evaluation = ROOT/'artifacts/native_visual_flow_learning_v1/evaluation'
    preparation = json.loads((preflight/'preparation.json').read_text())
    source = json.loads((preflight/'execution/codec.json').read_text())
    if len(source) != 8 or len(preparation['selected_cases']) != 4:
        raise ValueError('all eight original mode/sample reconstructions required')
    references = {}
    for variant in ('normal', 'copy'):
        for line in (evaluation/(variant+'_references.jsonl')).read_text().splitlines():
            row = json.loads(line)
            references[(row['id'],row['target_sha256'])] = row
    binary, evaluator = engine()
    output.mkdir(parents=True, exist_ok=False)
    (output/'images').mkdir()
    observed = []
    for i, case in enumerate(preparation['selected_cases']):
        target = data/case['target']
        if sha(target) != case['target_sha256']:
            raise ValueError('original target changed')
        ref = references[(case['id'],case['target_sha256'])]
        with Image.open(target) as image:
            if image.size != (1024,1024) or image.mode != 'RGB':
                raise ValueError('full original native answer target required')
            gold = np.asarray(image).copy()
        raw, diagnostic = recognize(binary, target)
        clean_exact = normalize(raw,ref['language']) == normalize(ref['reference_text'],ref['language'])
        mask = gold.mean(2) < 128
        for kind in ('mode','sample'):
            row = source[2*i+(kind=='sample')]
            if (row['id'],row['kind'],row['target_sha256']) != (case['id'],kind,case['target_sha256']):
                raise ValueError('complete original reconstruction order changed')
            original = Path(row['saved']['path'])
            signature = authenticate_native_png(original,row['saved']['sha256'],row['saved']['tensor_sha256'])
            with Image.open(original) as image:
                pixels = np.asarray(image).copy()
            ink = pixels.mean(2) < 128
            tp, fp, fn = int((mask&ink).sum()), int((~mask&ink).sum()), int((mask&~ink).sum())
            f1 = 2*tp/max(1,2*tp+fp+fn)
            ratio = int(ink.sum())/max(1,int(mask.sum()))
            if (f1 != row['foreground_F1'] or ratio != row['predicted_to_target_ink_ratio']
                    or int(mask.sum()) != row['target_black_pixels']):
                raise ValueError('original ink measurements do not reproduce')
            decoded_raw, decoded_diagnostic = recognize(binary, original)
            name = f'images/{i:02}_{kind}.png'
            shutil.copyfile(original,output/name)
            if sha(output/name) != row['saved']['sha256']:
                raise ValueError('archived image bytes differ')
            observed.append(dict(id=case['id'],kind=kind,language=ref['language'],
                target_sha256=case['target_sha256'],reference_text=ref['reference_text'],
                reference_raw_transcript=raw,reference_OCR_stderr=diagnostic,reference_OCR_exact=clean_exact,
                raw_transcript=decoded_raw,OCR_stderr=decoded_diagnostic,
                OCR_exact=normalize(decoded_raw,ref['language'])==normalize(ref['reference_text'],ref['language']),
                foreground_F1=f1,predicted_to_target_ink_ratio=ratio,target_black_pixels=int(mask.sum()),
                image=name,PNG_sha256=row['saved']['sha256'],actual_tensor_file_sha256=row['saved']['tensor_sha256'],
                actual_pixel_signature=signature))
    if torch.cuda.is_initialized():
        raise ValueError('offline audit initialized CUDA')
    save_json(output/'report.json',dict(rows=observed,complete_original_codec_cases=4,actual_saved_reconstructions=8,
        reference_OCR_exact=sum(r['reference_OCR_exact'] for r in observed[::2]),
        reconstructed_OCR_exact=sum(r['OCR_exact'] for r in observed),
        foreground_F1_range=[min(r['foreground_F1'] for r in observed),max(r['foreground_F1'] for r in observed)],
        engine=evaluator,implementation_sha256=sha(__file__),
        source_preparation_sha256=sha(preflight/'preparation.json'),source_codec_report_sha256=sha(preflight/'execution/codec.json'),
        OCR_implementation_sha256=sha(ROOT/'scripts/evaluate_image_edit_baseline.py'),
        new_codec_or_generator_calls=0,neural_models_loaded=0,all_saved_PNGs_authenticated=True,
        all_original_ink_measurements_reproduced=True,raw_image_review_completed=False,
        target_reconstruction_is_not_autonomous_generation=True,semantic_learning_validated=False,
        general_ilm_goal_complete=False))
    print('Authenticated all8 saved codec reconstructions; no model loaded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output.resolve())
