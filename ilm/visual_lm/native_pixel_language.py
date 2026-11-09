"""Native image reading, causal image history and normalized pixel writing.

This candidate retains full native prompt KV. Its suffix adapters never affect
prompt prefill. A frozen image codec encodes observed history only; its image
decoder and the previous image-code prediction flow are not output paths.
"""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .adaptive_image_continuation import AdaptiveImageLanguageModel
from .conditional_pixel_density import ConditionalPixelDensity, PixelDensityConfig, pack_pixels, unpack_pixels
from .native_image_continuation import NativeImagePrefix


class NativePixelLanguageModel(AdaptiveImageLanguageModel):
    def __init__(self, context, history_codec, *, input_scale, query_seed, raster_state):
        super().__init__(context, history_codec, input_scale=input_scale, query_seed=query_seed)
        del self.flow
        # The native readout is one 4096-state, not the earlier two-query cache.
        # Reset its condition projection and normalize each observed state.
        self.density = ConditionalPixelDensity(torch.zeros(4096), torch.ones(4096),
            PixelDensityConfig(condition_width=4096))
        state = self.density.state_dict()
        omitted = {'condition.weight', 'condition.bias', 'condition_mean', 'condition_scale'}
        transferred = {name:value for name,value in raster_state.items() if name not in omitted}
        if set(transferred) != set(state)-omitted:
            raise ValueError('the complete qualified physical decoder architecture is required')
        for name,value in transferred.items():
            if value.shape != state[name].shape or value.dtype != state[name].dtype:
                raise ValueError('physical decoder transfer shape/type mismatch')
            state[name].copy_(value)
        self.transferred_raster_tensors = len(transferred)
        self.density.load_state_dict(state)

    def normalized_conditions(self, states):
        if states.shape[-1] != 4096:
            raise ValueError('native 4096-dimensional contextual states required')
        return F.layer_norm(states.float(), (4096,))

    def warm_image_interfaces(self, state):
        """Reuse only the completed image-history encoder, scale and stop head."""
        accepted = {k:v for k,v in state.items() if k.startswith(('image_input.', 'stop.')) or k=='input_scale'}
        current = self.state_dict()
        expected = {k for k in current if k.startswith(('image_input.', 'stop.')) or k=='input_scale'}
        if set(accepted) != expected:
            raise ValueError('completed native image-interface state differs')
        with torch.no_grad():
            for name,value in accepted.items():
                if value.shape != current[name].shape or value.dtype != current[name].dtype:
                    raise ValueError('native image-interface transfer shape/type mismatch')
                current[name].copy_(value.to(current[name].device))

    def image_codes(self, images):
        if not isinstance(images, torch.Tensor) or images.ndim!=4 or images.shape[1:]!=(1,32,64):
            raise ValueError('complete physical image blocks [T,1,32,64] required')
        if not bool(((images==-1)|(images==1)).all()):
            raise ValueError('lossless signed binary image blocks required')
        return self.writer.encode_targets(images.to(self.device)).detach()

    def loss_from_codes(self, prefix, images, codes, *, writer_blocks=16, generator=None):
        """True past images may condition a block; its own image never does."""
        if images.ndim!=4 or images.shape[1:]!=(1,32,64) or not len(images):
            raise ValueError('a complete nonempty answer-image sequence is required')
        if codes.shape!=(len(images),2,768) or codes.requires_grad:
            raise ValueError('frozen observed-image codes must match the complete answer')
        if prefix.initial_state.shape[0]!=1 or type(writer_blocks) is not int or writer_blocks<1:
            raise ValueError('one owning prefix and a positive block-loss budget required')
        if not bool(((images==-1)|(images==1)).all()):
            raise ValueError('binary target images required')
        states=self.conditions(prefix,codes[None])
        count=len(images)
        chosen=(torch.arange(count,device=self.device) if count<=writer_blocks else
            torch.randperm(count,device=self.device,generator=generator)[:writer_blocks])
        target=pack_pixels(images[chosen,0].to(self.device)<0)
        logits=self.density(self.normalized_conditions(states[0,chosen]),target)
        image_nll=-F.log_softmax(logits,-1).gather(-1,target[:,:,None]).mean()
        stop_logits=self.stop(states.float()).squeeze(-1)
        stopping=.5*(F.softplus(stop_logits[0,:count]).mean()+F.softplus(-stop_logits[0,count]))
        # Equal image/block-mean and balanced-stop losses are an explicit
        # optimization weighting, not the unscaled full joint answer NLL.
        loss=image_nll+stopping
        return loss,dict(image_nll=image_nll.detach(),stop=stopping.detach(),
            sampled_blocks=len(chosen),complete_answer_blocks=count,
            selected_blocks=chosen.detach(),query_states=states.detach())

    def training_loss(self, prefix, images, *, writer_blocks=16, generator=None):
        return self.loss_from_codes(prefix,images,self.image_codes(images),
            writer_blocks=writer_blocks,generator=generator)

    @torch.no_grad()
    def emit_image(self, state):
        patterns=self.density.generate(self.normalized_conditions(state))
        ink=unpack_pixels(patterns)
        return torch.where(ink,-torch.ones_like(ink,dtype=torch.float32),
            torch.ones_like(ink,dtype=torch.float32))[:,None]

    @torch.no_grad()
    def append_emitted(self, current, image):
        """Read the actual emitted pixels, never the desired target image."""
        actual=self.image_codes(image)
        hidden,values=self.continuation(self.embed_images(actual[None]),current,
            checkpoint_layers=False,collect_cache=True)
        updated=NativeImagePrefix(values,hidden[:,-1:],current.rotary_offsets+2,current.blocked_before)
        return updated,hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest()

    @torch.no_grad()
    def generate(self, prefix, *, generator=None, maximum_blocks=None):
        cap=self.maximum_blocks if maximum_blocks is None else maximum_blocks
        if type(cap) is not int or not 1<=cap<=self.maximum_blocks:
            raise ValueError('a uniform physical output cap in [1,128] is required')
        current=self.start_prefix(prefix)
        images,probabilities,feedback=[],[],[]
        stopped=False
        for position in range(cap+1):
            condition=current.initial_state[:,0].float()
            stop=self.stop(condition).float().sigmoid()
            if not bool(torch.isfinite(stop).all()):
                raise FloatingPointError('nonfinite structural stop decision')
            probability=float(stop.item());probabilities.append(probability)
            if probability>=.5:
                stopped=True;break
            if position==cap:
                break
            image=self.emit_image(condition)
            current,digest=self.append_emitted(current,image)
            images.append(image.cpu());feedback.append(digest)
        return dict(images=torch.cat(images) if images else torch.empty(0,1,32,64),
            stopped=stopped,capped=not stopped,stop_probabilities=probabilities,
            feedback_sha256=feedback,feedback_from_emitted_pixels=True,
            native_context_kv_retained=True,tokenizer=False,language_vocabulary=False,
            persistent_glyph_inventory=False,adapted_writing_suffix=True,
            initial_read_query=True,direct_physical_pixel_output=True,
            codec_decoder_used_for_output=False)

    def trainable_state_dict(self):
        return {name:value.detach().cpu().clone() for name,value in self.state_dict().items()
            if name.startswith(('image_input.','density.','stop.'))
            or name in ('input_scale','start_query')
            or (name.startswith('continuation.context.') and name.endswith(('.down','.up')))}

    @torch.no_grad()
    def load_trainable_state_dict(self,state):
        current=self.trainable_state_dict()
        if state.keys()!=current.keys() or any(state[n].shape!=v.shape or state[n].dtype!=v.dtype for n,v in current.items()):
            raise ValueError('native pixel actor checkpoint architecture differs')
        target=self.state_dict()
        for name,value in state.items():target[name].copy_(value.to(target[name].device))
