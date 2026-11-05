"""Causal image-language actor with a continuous visual plan and fixed renderer.

The input reader and output renderer use separate continuous spaces. This
actor predicts output-codec coordinates from observed prompt/history pixels.
Future codec states occur only as detached supervised targets.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .codec_block_writer import CodecBlockWriter
from .grounded_image_actor import GroundedImageActor,ImageAdapterLinear,validate_image


@dataclass(frozen=True)
class CodecVisualLanguageLoss:
    state_weight: float = 1.
    image_weight: float = 1.
    stop_weight: float = 1.

    def __post_init__(self):
        if any(not math.isfinite(x) or x<=0 for x in
                (self.state_weight,self.image_weight,self.stop_weight)):
            raise ValueError("finite positive objective weights required")


def normalized_visual_state(value):
    if value.ndim!=3 or value.shape[1:]!=(2,768):
        raise ValueError("two continuous 768-dimensional output states required")
    return F.layer_norm(value.float(),(768,))


class ContinuousVisualCodeHead(nn.Module):
    """Predict continuous output-codec states from semantic contextual queries."""

    def __init__(self,device):
        super().__init__()
        self.projection=nn.Linear(4096,768,device=device,dtype=torch.float32)
        nn.init.zeros_(self.projection.bias)

    def forward(self,states):
        if states.ndim!=3 or states.shape[1:]!=(2,4096):
            raise ValueError("physical contextual queries must have shape [N,2,4096]")
        with torch.autocast(device_type=states.device.type,enabled=False):
            return F.layer_norm(self.projection(states.float()),(768,))


class CodecVisualLanguageActor(nn.Module):
    """Adapt the complete real contextual body to predict continuous writing.

    A qualified fixed CodecBlockWriter must be supplied by the caller. This
    class implements the next candidate; its existence is not a quality result.
    """

    def __init__(self,actor:GroundedImageActor,loss=CodecVisualLanguageLoss()):
        super().__init__()
        if not isinstance(actor,GroundedImageActor) or not isinstance(actor.writer,CodecBlockWriter):
            raise TypeError("complete native actor with a qualified codec output required")
        if actor.context.config.vocab_size!=0 or actor.context.embed_tokens is not None:
            raise ValueError("the contextual body must have no language vocabulary")
        self.actor=actor.requires_grad_(False)
        self.loss_config=loss
        for layer in self.actor.context.layers:
            for name in ("q_proj","v_proj","o_proj"):
                adapter=getattr(layer.self_attn,name)
                if not isinstance(adapter,ImageAdapterLinear):
                    raise TypeError("the verified contextual adapters are required")
                adapter.down.requires_grad_(True)
                adapter.up.requires_grad_(True)
        self.actor.queries.requires_grad_(True)
        self.actor.stop.requires_grad_(True)
        self.bridge=ContinuousVisualCodeHead(self.actor.device)
        self.actor.vision.eval()
        self.actor.writer.eval()

    @property
    def device(self):
        return self.actor.device

    def train(self,mode=True):
        super().train(mode)
        self.actor.vision.eval()
        self.actor.writer.eval()
        return self

    def stop_logits(self,states):
        with torch.autocast(device_type=self.device.type,enabled=False):
            return self.actor.stop(states.float().mean(dim=1)).flatten().float()

    def training_loss(self,prompt,targets,*,generator,writer_blocks=16):
        """Train on complete causal history; targets never condition their query.

        Full target images are permitted in teacher forcing. Independent image
        encoding and the existing sequential contextual mask restrict each
        query to the prompt and completed blocks. Teacher visual states are
        detached labels; generation has no access to this label path.
        """
        validate_image(prompt)
        if not targets or type(writer_blocks) is not int or writer_blocks<1:
            raise ValueError("actual answer images and positive loss sample count required")
        for image in targets:
            validate_image(image)
            if image.shape!=(1,1,32,64):
                raise ValueError("physical target blocks must be 32 by 64")
        features,deep,grids=self.actor.encode_images([prompt,*targets])
        inputs,queries,_=self.actor.assemble(features,deep,grids)
        with torch.autocast(device_type=self.device.type,enabled=False):
            output=self.actor.context(**inputs,use_cache=False)
        positions=torch.tensor([[i,i+1] for i in queries],device=self.device)
        states=output.last_hidden_state[0,positions]
        chosen=torch.randperm(len(targets),device=self.device,generator=generator)[:writer_blocks]
        plan=self.bridge(states[chosen])
        target=torch.cat([targets[i].to(self.device) for i in chosen.tolist()])
        teacher=self.actor.writer.encode_targets(target).detach()
        state_loss=F.mse_loss(normalized_visual_state(plan),normalized_visual_state(teacher))
        with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16):
            prediction=self.actor.writer.predict(plan)
        image_loss=F.mse_loss(prediction.float(),target)
        logits=self.stop_logits(states)
        stop_loss=.5*(F.softplus(-logits[-1])+F.softplus(logits[:-1]).mean())
        weights=self.loss_config
        total=weights.state_weight*state_loss+weights.image_weight*image_loss+weights.stop_weight*stop_loss
        return total,dict(state=state_loss.detach(),image=image_loss.detach(),stop=stop_loss.detach(),
            selected_blocks=chosen.detach(),query_states=states.detach(),
            predicted_visual_states=plan.detach(),teacher_visual_states=teacher.detach())

    @torch.no_grad()
    def generate(self,prompt,*,maximum_blocks=128):
        """Read observed pixels and then reread only generated physical blocks."""
        if type(maximum_blocks) is not int or maximum_blocks<1:
            raise ValueError("a positive physical output capacity is required")
        validate_image(prompt)
        self.eval()
        states,past,offset=self.actor.conditions(prompt,use_cache=True)
        condition=states[-1:]
        images,events=[],[]
        for index in range(maximum_blocks):
            logits=self.stop_logits(condition)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("nonfinite structural stop prediction")
            probability=float(logits.sigmoid())
            if probability>=.5:
                events.append(dict(block=index,stop_probability=probability,stopped=True))
                break
            plan=self.bridge(condition)
            with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16):
                pixels=self.actor.writer.sample(plan,steps=1)
            if not bool(torch.isfinite(pixels).all()):
                raise FloatingPointError("nonfinite physical writing prediction")
            image=torch.where(pixels>=0,1.,-1.).cpu()
            images.append(image)
            events.append(dict(block=index,stop_probability=probability,stopped=False))
            if index+1<maximum_blocks:
                condition,past,offset=self.actor.append_generated(image,past,offset=offset)
        return dict(images=tuple(images),events=events,stopped=bool(events[-1]["stopped"]),
            capped=not events[-1]["stopped"])

    def trainable_state(self):
        return {name:p.detach().cpu().clone() for name,p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self,state):
        parameters={name:p for name,p in self.named_parameters() if p.requires_grad}
        if parameters.keys()!=state.keys():
            raise ValueError("trainable continuous actor identity differs")
        with torch.no_grad():
            for name,parameter in parameters.items():
                if parameter.shape!=state[name].shape:
                    raise ValueError("trainable continuous actor shape differs")
                parameter.copy_(state[name])
