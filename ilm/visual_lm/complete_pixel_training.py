"""Variable-length, full-answer image likelihood over owning native prefixes."""
import torch
from torch.nn import functional as F

from .conditional_pixel_density import pack_pixels


def complete_pixel_loss(model, prefix, images, codes, lengths, *, checkpoint_layers=True):
    """Mean per-question image NLL plus balanced continue/end supervision.

    All physical target pieces are scored. Padded history after each answer
    cannot affect its valid queries. No current answer block conditions itself.
    The disclosed question/block means are not the unscaled joint stream NLL.
    """
    if (not isinstance(images,(list,tuple)) or len(images)!=len(codes)
            or lengths.shape!=(len(codes),) or lengths.dtype!=torch.long
            or lengths.device!=model.device or codes.device!=model.device
            or codes.ndim!=4 or codes.shape[2:]!=(2,768) or codes.requires_grad
            or bool(((lengths<1)|(lengths>codes.shape[1])).any())
            or len(prefix.initial_state)!=len(codes)):
        raise ValueError('complete owning images, frozen history codes and lengths required')
    counts=lengths.tolist()
    for n,image in zip(counts,images):
        if (image.shape!=(n,1,32,64) or not image.is_floating_point()
                or not bool(((image==-1)|(image==1)).all())):
            raise ValueError('each complete target must contain its exact binary blocks')
    states=model.conditions(prefix,codes,checkpoint_layers=checkpoint_layers)
    valid=torch.cat([states[b,:n] for b,n in enumerate(counts)])
    target=pack_pixels(torch.cat([image.to(model.device) for image in images])[:,0]<0)
    logits=model.density(model.normalized_conditions(valid),target)
    per_block=-logits.log_softmax(-1).gather(-1,target[:,:,None]).squeeze(-1).mean(-1)
    image_nll=torch.stack([part.mean() for part in per_block.split(counts)]).mean()
    stop_logits=model.stop(states.float()).squeeze(-1)
    stopping=torch.stack([.5*(F.softplus(stop_logits[b,:n]).mean()+F.softplus(-stop_logits[b,n]))
        for b,n in enumerate(counts)]).mean()
    return image_nll+stopping,dict(image_nll=image_nll.detach(),stop=stopping.detach(),
        complete_blocks=sum(counts),questions=len(counts),all_target_pieces_scored=True)


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu_tree(v) for v in value)
    return value
