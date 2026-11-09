"""Load learned pixel-language state without earlier training checkpoints.

The deployed actor requires the authenticated frozen body, history encoder,
numeric controls and its own learned state. No training examples are read.
"""
import gc
import hashlib
import json
from pathlib import Path

import torch

from .conditional_pixel_density import PixelDensityConfig
from .fp64_pixel_density import FP64ConditionalPixelDensity
from .native_image_actor import build_native_image_actor
from .native_pixel_language import NativePixelLanguageModel


ROOT=Path(__file__).resolve().parents[2]
RUNTIME_CORE=tuple('ilm/visual_lm/'+name+'.py' for name in (
    'native_pixel_language','fp64_pixel_density','conditional_pixel_density',
    'adaptive_image_continuation','native_image_continuation','native_image_actor',
    'image_context_interface','semantic_image_memory','grounded_image_actor','grounded_image_weights',
    'gguf_visual_language','gguf_language_trunk','codec_block_writer','continuous_glyph_codec',
    'suffix_attention_mask','clean_writing_flow','semantic_writing_model'))


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):value.update(part)
    return value.hexdigest()


def read_pixel_checkpoint(checkpoint, training_run, *, allow_incomplete=False):
    """Validate owning preparation and physical-head schema on CPU only."""
    checkpoint,training_run=Path(checkpoint),Path(training_run)
    preparation=training_run/'preparation.json'
    manifest=json.loads(preparation.read_text())
    # Deployment binds runtime equations, not an old training script or data.
    for name in RUNTIME_CORE:
        if manifest['code_sha256'].get(name)!=sha(ROOT/name):raise ValueError('checkpoint runtime source differs: '+name)
    with checkpoint.open('rb') as stream:
        digest=hashlib.sha256()
        for part in iter(lambda:stream.read(8*1024*1024),b''):digest.update(part)
        stream.seek(0);saved=torch.load(stream,map_location='cpu',weights_only=True)
    if (saved.get('preparation_sha256')!=sha(preparation) or type(saved.get('update')) is not int
            or saved['update']<1 or not isinstance(saved.get('model'),dict)):
        raise ValueError('a source-bound learned pixel checkpoint is required')
    finished=saved.get('training_complete',False)
    if not finished and not allow_incomplete:
        raise ValueError('checkpoint is an unfinished research milestone; use --allow-incomplete explicitly or wait for training to finish')
    state=saved['model']
    if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('finite FP32 pixel actor state required')
    head=FP64ConditionalPixelDensity(torch.zeros(4096),torch.ones(4096),PixelDensityConfig(condition_width=4096))
    raster={name[len('density.'):]:v for name,v in state.items() if name.startswith('density.')}
    head.load_state_dict(raster,strict=True)
    expected={f'density.{name}':value.shape for name,value in head.state_dict().items()}
    expected.update({'start_query':(1,1,4096),'stop.weight':(1,4096),'stop.bias':(1,),
        'image_input.0.weight':(768,),'image_input.0.bias':(768,),
        'image_input.1.weight':(1024,768),'image_input.1.bias':(1024,),
        'image_input.3.weight':(4096,1024),'image_input.3.bias':(4096,),'input_scale':()})
    for layer in range(36):
        for name,out_features in (('q_proj',4096),('v_proj',1024),('o_proj',4096)):
            prefix=f'continuation.context.layers.{layer}.self_attn.{name}'
            expected[prefix+'.down']=(8,4096);expected[prefix+'.up']=(out_features,8)
    if set(state)!=set(expected):raise ValueError('incomplete or incompatible native actor parameter keys')
    for name,shape in expected.items():
        if state[name].shape!=shape:raise ValueError('incompatible native pixel state: '+name)
    if float(state['input_scale'])<=0 or bool((state['density.condition_scale']<=0).any()):
        raise ValueError('positive native input and conditioning scales required')
    identity=dict(checkpoint_sha256=digest.hexdigest(),checkpoint_update=saved['update'],
        checkpoint_training_finished=finished,preparation_sha256=sha(preparation),
        external_pretraining=True,language_quality_claim=False)
    return state,raster,identity


def load_pixel_actor(state,raster, *, controls,codec,ollama_root,gguf_source,device):
    """Build from frozen source weights and restore only this actor's state."""
    actor,provenance=build_native_image_actor(controls=controls,codec=codec,ollama_root=ollama_root,
        gguf_source=gguf_source,device=device)
    previous=actor.writing
    writer=NativePixelLanguageModel(previous.continuation.context,previous.writer,
        input_scale=float(state['input_scale']),query_seed=0,raster_state=raster).to(device)
    writer.load_trainable_state_dict(state);actor.writing=writer
    del previous;gc.collect();torch.cuda.empty_cache()
    actor.requires_grad_(False).eval()
    def forbidden(*args,**kwargs):raise RuntimeError('a legacy image-code decoder entered pixel-language output')
    writer.writer.predict=forbidden;writer.writer.codec.decode=forbidden
    return actor,dict(**provenance,own_learned_state_restored=True,legacy_training_checkpoint_required=False,
        legacy_raster_checkpoint_required=False,physical_head_arithmetic='float64',parameter_storage='float32')
