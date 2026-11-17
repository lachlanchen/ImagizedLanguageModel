"""Uniform physical-form support for codec learning, independent of word rate."""
import torch


def balanced_codec_support(partition,seed):
    group=partition['group'];train=partition['train_index'];held=partition['held']
    if group.ndim!=1 or held.shape!=group.shape or train.ndim!=1 or bool(held[train].any()):
        raise ValueError('Complete training-only physical partition required')
    counts=torch.bincount(group[train],minlength=len(partition['group_mask_sha256']))
    first={}
    for index in train.tolist():first.setdefault(int(group[index]),index)
    representatives=torch.tensor(sorted(first.values()),dtype=torch.long)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    order=representatives[torch.randperm(len(representatives),generator=generator)]
    weights=torch.zeros(len(group),dtype=torch.float64)
    weights[train]=len(train)/(len(representatives)*counts[group[train]].double())
    return dict(order=order,representatives=representatives,complete_source_weights=weights,
                training_forms=len(representatives),training_cells=len(train))
