"""End-to-end contextual learning with the existing complete visual-span model.

Both matched arms use FP32 parameters and the same full-rank writer. The
treatment makes the contextual body trainable; the comparison freezes it.
No vocabulary embedding or lexical output tensor enters either model.
"""
import torch

from .compact_visual_span_weights import load_compact_visual_span_model
from .compact_visual_span_training import CompactVisualSpanObjective
from .full_rank_visual_span import TRAINING_SEED, widen_writer
from .offloaded_adamw import OffloadedAdamW


def build_model(local, codec, device, *, seed=TRAINING_SEED):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model,provenance=load_compact_visual_span_model(local,codec,device)
    objective=CompactVisualSpanObjective().to(device)
    model.writer=widen_writer(model.writer,seed)
    model.context.float()
    model.context.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train();model.codec.requires_grad_(False).eval()
    if model.context.embed_tokens is not None:raise ValueError('Lexical input table entered visual model')
    if any('lm_head' in n or 'embed_tokens' in n for n in model.state_dict()):
        raise ValueError('Lexical head/table entered visual model state')
    provenance.update(context_parameter_precision='float32',writer_width=1024,
        context_parameters=sum(p.numel() for p in model.context.parameters()),
        total_model_parameters=sum(p.numel() for p in model.parameters()),
        teacher_projection_parameters=sum(p.numel() for p in objective.parameters()),
        source_pretraining_inherited=True,language_capability_proved=False)
    return model,objective,provenance


def configure_optimizer(model, objective, *, train_context, guard=lambda:None):
    model.context.requires_grad_(train_context)
    model.codec.requires_grad_(False)
    groups=[dict(params=[*model.encoder.parameters(),*model.writer.parameters(),*objective.parameters()],lr=2e-4)]
    if train_context:groups.append(dict(params=list(model.context.parameters()),lr=2e-5))
    optimizer=OffloadedAdamW(groups,lr=2e-4,weight_decay=.01,guard=guard)
    return optimizer
