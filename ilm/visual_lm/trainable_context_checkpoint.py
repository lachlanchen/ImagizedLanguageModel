"""Full learned-context snapshots; the frozen arm reuses its authenticated source."""
import torch

from .trainable_visual_context import build_model

SCHEMA='trainable-visual-context-deploy-v1'


def payload(model, identity, updates):
    def cpu(module):
        return {n:v.detach().cpu() for n,v in module.state_dict().items()}
    return dict(schema=SCHEMA,identity=identity,updates=updates,
                context=cpu(model.context) if identity['train_context'] else None,
                image_heads=dict(encoder=cpu(model.encoder),writer=cpu(model.writer)))


def restore(model, saved, identity):
    if set(saved)!={'schema','identity','updates','context','image_heads'} or saved['schema']!=SCHEMA:
        raise ValueError('Require a complete deployable visual-context checkpoint')
    if saved['identity']!=identity or type(saved['updates']) is not int or saved['updates']<0:
        raise ValueError('Checkpoint identity or update counter differs')
    if identity['train_context']:
        if not isinstance(saved['context'],dict) or not saved['context']:
            raise ValueError('Learned context tensors are mandatory')
        if any(v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in saved['context'].values()):
            raise ValueError('Keep complete finite FP32 learned context')
        model.context.load_state_dict(saved['context'],strict=True)
    elif saved['context'] is not None:
        raise ValueError('Frozen control must reuse the authenticated original context')
    for name in ('encoder','writer'):
        getattr(model,name).load_state_dict(saved['image_heads'][name],strict=True)


def load_deploy(path, local, codec, device):
    from ilm.image_edit.baseline import ROOT,sha
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    identity=saved['identity']
    if sha(codec)!=identity['codec_sha256']:
        raise ValueError('Codec differs from training')
    for name,digest in identity['model_sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Bound model source changed: '+name)
    model,objective,provenance=build_model(local,codec,device)
    del objective
    restore(model,saved,identity)
    provenance.pop('trainable_image_parameters',None)
    provenance.pop('visual_span_heads_initialized_for_training',None)
    provenance.update(learned_context_restored=identity['train_context'],
                      total_model_parameters=sum(p.numel() for p in model.parameters()),
                      checkpoint_schema=SCHEMA)
    model.context.gradient_checkpointing_disable()
    return model.requires_grad_(False).eval(),saved['updates'],provenance
