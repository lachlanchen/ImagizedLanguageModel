#!/usr/bin/env python3
"""Read a question image and write an autonomous glyph-image answer.

Inference needs learned neural weights, numeric task controls and the shared
frozen visual/contextual body. It never opens a training corpus, font, glyph
inventory, source tokenizer or source input/output vocabulary table.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.score_pixel_language_first_blocks import diagnostic_gpu,require
from scripts.train_semantic_glyph_language import Monitor,numerics,atomic_tensor

CONTROL_SHA='e53cb8d13892aa5de283d55138101ca45527356f73b40a9a965cee77c67167c1'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/semantic_glyph_language_v1/fit/checkpoint_latest.pt')
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--controls',type=Path,default=ROOT/'artifacts/native_visual_transfer_v71_20261105/attempt01/structural_vectors.pt')
    parser.add_argument('--maximum-glyphs',type=int,default=64)
    args=parser.parse_args()
    require(not args.output.exists(),'choose a new output directory')
    require(1<=args.maximum_glyphs<=256,'output safety cap must be1..256glyphs')
    require(sha(args.controls)==CONTROL_SHA,'native numeric task controls changed')
    digest=sha(args.checkpoint)
    if args.expected_checkpoint_sha256:require(digest==args.expected_checkpoint_sha256,'checkpoint digest differs')
    # Host image preparation has no language interpretation.
    source=Image.open(args.prompt_image).convert('RGBA')
    white=Image.new('RGBA',source.size,'white');white.alpha_composite(source)
    source=white.convert('L');original_size=list(source.size)
    if source.width*source.height>1048576:
        ratio=(1048576/(source.width*source.height))**.5
        source=source.resize((max(1,int(source.width*ratio)),max(1,int(source.height*ratio))),Image.Resampling.BILINEAR)
    canvas=Image.new('L',(max(32,(source.width+31)//32*32),max(32,(source.height+31)//32*32)),255)
    canvas.paste(source,(0,0))
    args.output.mkdir(parents=True);canvas.save(args.output/'observed_prompt.png')
    monitor=Monitor(args.output,1200.);monitor.status('admitting')
    try:
        import torch
        from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge,SemanticGlyphLanguageActor
        from ilm.visual_lm.grounded_image_weights import load_grounded_image_actor
        from ilm.visual_lm.image_context_interface import ImageContextInterface
        with diagnostic_gpu() as (device,admission),torch.no_grad():
            monitor.admission=admission;numerics();monitor.guard()
            saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
            require(saved.get('language_training_complete') is True,'primitive-only checkpoint cannot answer')
            bridge=SemanticGlyphBridge().to(device);bridge.load_state_dict(saved['model'],strict=True)
            bridge.requires_grad_(False).eval();del saved
            local=ROOT.parent/'LocalLLM/.local'
            base,provenance=load_grounded_image_actor(ollama_root=local/'models/ollama',
                gguf_source=local/'tools/llama.cpp/gguf-py',device=device)
            base.writer=torch.nn.Identity();base.requires_grad_(False).eval()
            interface=ImageContextInterface(base,**torch.load(args.controls,map_location='cpu',weights_only=True))
            actor=SemanticGlyphLanguageActor(interface,bridge,language_training_complete=True).eval()
            hook=base.context.register_forward_pre_hook(lambda module,inputs:monitor.guard())
            prompt=torch.from_numpy(np.asarray(canvas).copy()).float()[None,None]/127.5-1
            value=actor.generate(prompt,maximum_glyphs=args.maximum_glyphs)
            hook.remove();atomic_tensor(args.output/'answer.pt',value)
            tiles=value['images'];array=np.full((32,max(1,len(tiles))*32),255,dtype=np.uint8)
            for i,tile in enumerate(tiles):array[:,i*32:(i+1)*32]=((tile[0].numpy()+1)*127.5).astype(np.uint8)
            Image.fromarray(array).save(args.output/'answer.png')
            save_json(args.output/'report.json',dict(checkpoint_sha256=digest,controls_sha256=CONTROL_SHA,
                original_prompt_sha256=sha(args.prompt_image),observed_prompt_sha256=sha(args.output/'observed_prompt.png'),
                original_size=original_size,observed_size=list(canvas.size),glyphs=len(tiles),
                stopped=value['stopped'],capped=value['capped'],maximum_glyphs=args.maximum_glyphs,
                answer_sha256=sha(args.output/'answer.pt'),answer_PNG_sha256=sha(args.output/'answer.png'),
                runtime_font=False,runtime_tokenizer=False,runtime_training_corpus=False,
                runtime_glyph_inventory=False,source_vocabulary_parameters=False,external_teacher=False,
                actual_pixels_feedback=True,known_general_language_reliability=False,
                provenance=provenance,**monitor.receipt()))
            monitor.status('completed')
            print(json.dumps(dict(answer=str(args.output/'answer.png'),glyphs=len(tiles),
                                  stopped=value['stopped'],capped=value['capped'])))
    except BaseException as error:
        monitor.status('failed',reason=repr(error));raise
    finally:
        if (args.output/'status.json').exists():
            save_json(args.output/'status.json',dict(json.loads((args.output/'status.json').read_text()),pid=None))


if __name__=='__main__':main()
