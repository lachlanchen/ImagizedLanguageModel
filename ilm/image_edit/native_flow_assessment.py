"""Offline authentication and fixed developmental criteria for native flow."""
import hashlib
from pathlib import Path

from .baseline import sha


def authenticate_native_png(path, png_sha256, tensor_sha256):
    """Authenticate actual decoded pixels using the source conversion exactly."""
    import numpy as np
    import torch
    from PIL import Image
    path=Path(path);tensor=path.parent/'actual_tensors.pt'
    if sha(path)!=png_sha256 or sha(tensor)!=tensor_sha256:
        raise ValueError('actual generation image/tensor bytes changed')
    value=torch.load(tensor,map_location='cpu',weights_only=True)
    pixels=value['pixels']
    if pixels.shape!=(1,3,1024,1024) or not torch.isfinite(pixels).all():
        raise ValueError('complete finite native decoded page required')
    expected=torch.clamp(127.5*pixels[0].permute(1,2,0)+128.,0,255).to(torch.uint8).numpy()
    with Image.open(path) as image:
        if image.mode!='RGB' or not np.array_equal(expected,np.asarray(image)):
            raise ValueError('saved PNG differs from actual native decoder tensor')
    return dict(dtype=str(pixels.dtype),shape=list(pixels.shape),
        decoded_pixel_bytes_sha256=hashlib.sha256(pixels.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())


def summarize(rows, *, paired):
    result={}
    for split in ('train','validation'):
        selected=[r for r in rows if r['split']==split]
        record=dict(cases=len(selected),canonical_exact=sum(r['canonical_exact'] for r in selected),
            accepted_exact=sum(r['accepted_exact'] for r in selected))
        if paired:
            pairs={}
            for row in selected:pairs.setdefault(row['pair'],[]).append(row)
            if any(len(group)!=2 for group in pairs.values()):raise ValueError('complete same-passage pairs required')
            record.update(pairs=len(pairs),both_correct_pairs=sum(all(r['accepted_exact'] for r in group) for group in pairs.values()))
        result[split]=record
    return result


def developmental_gates(summaries):
    before=summaries['initial']['normal']['validation'];after=summaries['final']['normal']['validation']
    return dict(seen_fit=summaries['final']['normal']['train']['accepted_exact']>=24,
        held_answers=after['accepted_exact']>=max(12,before['accepted_exact']+4),
        held_pairs=after['both_correct_pairs']>=max(4,before['both_correct_pairs']+2),
        question_dependence=after['accepted_exact']>=summaries['final']['question_erased']['validation']['accepted_exact']+4,
        passage_dependence=after['accepted_exact']>=summaries['final']['passage_erased']['validation']['accepted_exact']+4,
        copying_retention=summaries['final']['copy']['validation']['canonical_exact']>=
            summaries['initial']['copy']['validation']['canonical_exact']-2)
