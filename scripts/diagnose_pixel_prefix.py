#!/usr/bin/env python3
"""Post-hoc correct-prefix diagnosis, never an autonomous language benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.train_conditional_pixel_density import unchanged


def diagnose(source, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('hide CUDA for the offline prefix diagnosis')
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    import subprocess
    committed = subprocess.check_output(['git','show','HEAD:'+relative],cwd=ROOT)
    if hashlib.sha256(committed).hexdigest() != sha(__file__):
        raise ValueError('commit diagnostic source first')
    previous = unchanged(source)
    old = json.loads((source/'execution/report.json').read_text())
    if sha(source/'execution/final.pt') != old['final_checkpoint_sha256']:
        raise ValueError('owning checkpoint changed')
    if sha(source/'execution/ink.npy') != old['output_sha256']:
        raise ValueError('original autonomous outputs changed')
    import torch
    from torch.nn import functional as F
    from ilm.visual_lm.conditional_pixel_density import ConditionalPixelDensity, unpack_pixels
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    data = torch.load(source/'data.pt', map_location='cpu', weights_only=True)
    checkpoint = torch.load(source/'execution/final.pt', map_location='cpu', weights_only=True)
    model = ConditionalPixelDensity(data['mean'], data['scale'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    features = torch.cat((data['normal'], data['blank']))
    target = torch.cat((data['patterns'], data['patterns'][512:]))
    reference = torch.cat((data['ink'], data['ink'][512:])).numpy()
    auto = np.load(source/'execution/ink.npy', allow_pickle=False)
    predictions, losses = [], []
    with torch.no_grad():
        for i in range(0, 896, 32):
            logits = model(features[i:i+32], target[i:i+32])
            predictions.append(logits.argmax(-1))
            losses.append(-F.log_softmax(logits, -1).gather(-1, target[i:i+32, :, None]).squeeze(-1).mean(1))
    prediction, nll = torch.cat(predictions), torch.cat(losses)
    predicted_ink = unpack_pixels(prediction).numpy()
    cases = json.loads((source/'cases.json').read_text())
    rows = []
    for i, (image, ref) in enumerate(zip(predicted_ink, reference)):
        tp, fp, fn = int((image & ref).sum()), int((image & ~ref).sum()), int((~image & ref).sum())
        denominator = 2*tp+fp+fn
        j = i if i < 704 else i-192
        rows.append(dict(index=i, source=cases[j]['source'], split=cases[j]['split'],
            route='normal' if i < 704 else 'blank',
            correct_prefix_exact=bool(np.array_equal(image, ref)),
            correct_prefix_foreground_f1=2*tp/denominator if denominator else 1.,
            autonomous_exact=bool(np.array_equal(auto[i], ref)),
            correct_prefix_piece_accuracy=float((prediction[i] == target[i]).float().mean()),
            correct_prefix_nll=float(nll[i])))
    groups = {}
    for split in ('train', 'development'):
        for language in ('cmrc', 'dolly'):
            for route in (('normal',) if split == 'train' else ('normal', 'blank')):
                part = [r for r in rows if (r['split'], r['source'], r['route']) == (split, language, route)]
                groups[f'{split}_{language}_{route}'] = dict(count=len(part),
                    correct_prefix_exact=sum(r['correct_prefix_exact'] for r in part),
                    autonomous_exact=sum(r['autonomous_exact'] for r in part),
                    correct_prefix_foreground_f1=float(np.mean([r['correct_prefix_foreground_f1'] for r in part])),
                    correct_prefix_piece_accuracy=float(np.mean([r['correct_prefix_piece_accuracy'] for r in part])),
                    correct_prefix_nll=float(np.mean([r['correct_prefix_nll'] for r in part])))
    output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(predicted_patterns=prediction, supplied_prefix_nll=nll), output/'predictions.pt')
    from PIL import Image
    images = output/'images'
    images.mkdir()
    files = {}
    for i in (0, 256, 512, 640):
        for name, array in [('reference', reference[i]), ('autonomous', auto[i]), ('correct_prefix', predicted_ink[i])]:
            path = images/f'{i:04}_{name}.png'
            Image.fromarray((~array).astype('uint8')*255).save(path)
            files[str(path.relative_to(output))] = sha(path)
    save_json(output/'report.json', dict(groups=groups, cases=rows, actual_correct_prefix_forwards=896,
        source_preparation_sha256=sha(source/'preparation.json'), source_execution_report_sha256=sha(source/'execution/report.json'),
        source_checkpoint_sha256=old['final_checkpoint_sha256'], code_sha256=sha(__file__),
        predictions_sha256=sha(output/'predictions.pt'), files_sha256=files,
        protocol_sha256=sha(ROOT/'references/pixel_prefix_diagnosis_plan.md'),
        posthoc_diagnostic=True, current_target_piece_in_model_input=False,
        true_preceding_target_pieces_supplied=True, optimizer_updates=0, full_answers_generated=False,
        independent_semantic_accuracy=False, general_ilm_goal_complete=False))
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.source.resolve(), args.output.resolve())
