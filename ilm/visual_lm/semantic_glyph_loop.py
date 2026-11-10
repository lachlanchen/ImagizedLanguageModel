"""Continuous semantic glyph motor loop: image -> state -> glyph -> image.

The runtime has no character inventory, text tokenizer, source vocabulary
projection or font renderer. Training-only semantic targets live in datasets.
"""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F


def tensor_hash(value):
    raw=value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class SemanticGlyphBridge(nn.Module):
    def __init__(self, semantic_width=4096):
        super().__init__()
        self.semantic_width=semantic_width
        self.register_buffer('semantic_mean',torch.zeros(semantic_width))
        self.register_buffer('semantic_scale',torch.ones(semantic_width))
        self.encoder=nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),nn.GELU(),
            nn.Conv2d(64,128,4,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(128,256,4,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(256,256,4,stride=2,padding=1),nn.GELU(),nn.Flatten(),
            nn.Linear(4096,semantic_width))
        self.decoder_input=nn.Sequential(nn.LayerNorm(semantic_width),nn.Linear(semantic_width,2048),
            nn.GELU(),nn.Linear(2048,1024),nn.GELU())
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(64,128,4,stride=2,padding=1),nn.GELU(),
            nn.ConvTranspose2d(128,64,4,stride=2,padding=1),nn.GELU(),
            nn.ConvTranspose2d(64,32,4,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(32,1,3,padding=1))
        self.readout=nn.Sequential(nn.LayerNorm(semantic_width),nn.Linear(semantic_width,2048),
            nn.GELU(),nn.Linear(2048,semantic_width))
        self.stop=nn.Sequential(nn.LayerNorm(semantic_width),nn.Linear(semantic_width,256),
            nn.GELU(),nn.Linear(256,1))

    def normalize_semantic(self,value):
        return (value.float()-self.semantic_mean)/self.semantic_scale

    def denormalize_semantic(self,value):
        return value.float()*self.semantic_scale+self.semantic_mean

    def encode(self,images):
        if images.ndim!=4 or images.shape[1:]!=(1,32,32):
            raise ValueError('complete32x32 glyph images required')
        return self.encoder(images.float()).float()

    def decode_logits(self,semantic):
        hidden=self.decoder_input(semantic.float()).reshape(-1,64,4,4)
        return self.decoder(hidden).float()

    def feedback_embedding(self,images):
        return self.denormalize_semantic(self.encode(images))

    def predict(self,state):
        if state.ndim!=2 or state.shape[-1]!=self.semantic_width:
            raise ValueError('one native contextual vector per output glyph required')
        semantic=self.readout(state.float())
        return self.decode_logits(semantic),semantic,self.stop(state.float()).flatten()


class SemanticGlyphLanguageActor(nn.Module):
    """Native frozen visual reader with an independently trained glyph loop."""
    def __init__(self,interface,bridge,*,language_training_complete=False):
        super().__init__()
        if interface.actor.context.embed_tokens is not None:
            raise ValueError('a lexical embedding table entered the image runtime')
        self.interface,self.bridge=interface,bridge
        self.language_training_complete=language_training_complete

    @property
    def device(self):
        return self.interface.device

    @torch.no_grad()
    def generate(self,prompt,*,maximum_glyphs=256,retain_states=True):
        if not self.language_training_complete:
            raise ValueError('glyph reconstruction pretraining is not a language checkpoint')
        if not isinstance(maximum_glyphs,int) or maximum_glyphs<1:
            raise ValueError('positive autonomous output cap required')
        from .semantic_image_memory import native_budget_image
        arguments,geometry=self.interface.assemble(native_budget_image(prompt))
        output=self.interface.actor.context(**arguments,use_cache=True)
        state,past=output.last_hidden_state[:,-1:],output.past_key_values
        del output,arguments
        offset=geometry['rotary_offset']
        images,probabilities,feedback,states=[],[],[],[]
        stopped=False
        for i in range(maximum_glyphs+1):
            if retain_states: states.append(state[0,-1].float().cpu())
            logits,semantic,stop=self.bridge.predict(state[:,-1].float())
            probability=float(stop.sigmoid()[0]); probabilities.append(probability)
            if probability>=.5:
                stopped=True
                break
            if i==maximum_glyphs:
                break
            image=1-2*(logits.sigmoid()>=.5).float()
            images.append(image.detach().cpu())
            # Only the actually emitted pixels determine the next embedding.
            embedded=self.bridge.feedback_embedding(image)
            feedback.append(dict(image_sha256=tensor_hash(image),embedding_sha256=tensor_hash(embedded)))
            positions=torch.tensor([past.get_seq_length(),offset,offset,offset],device=self.device).reshape(4,1,1)
            output=self.interface.actor.context(inputs_embeds=embedded[:,None],position_ids=positions,
                past_key_values=past,use_cache=True)
            state,past=output.last_hidden_state[:,-1:],output.past_key_values
            del output,logits,semantic,embedded
            offset+=1
        return dict(images=torch.cat(images) if images else torch.empty(0,1,32,32),
            stop_probabilities=probabilities,stopped=stopped,capped=not stopped,
            feedback=feedback,states=torch.stack(states) if states else torch.empty(0,self.bridge.semantic_width),
            image_input_image_output=True,tokenizer=False,language_vocabulary=False,
            font_lookup=False,external_teacher=False,feedback_from_actual_emitted_pixels=True,
            learned_initial_query=False,native_initial_semantic_state=True,observer_geometry=geometry)


def reconstruction_loss(logits,images):
    ink=(1-images.float())/2
    raw=F.binary_cross_entropy_with_logits(logits,ink,reduction='none')
    black=ink.sum().clamp_min(1)
    white=(1-ink).sum().clamp_min(1)
    return .5*((raw*ink).sum()/black+(raw*(1-ink)).sum()/white)
