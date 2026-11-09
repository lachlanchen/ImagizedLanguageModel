#!/usr/bin/env python3
"""Typed, image or mixed prompts to direct learned answer-image streams."""
import argparse
import json
from pathlib import Path
import time

from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT,gpu_zero,save_json,sha
from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.codec_image_interface import prepare_image_prompt,save_prepared_prompt
from ilm.visual_lm.complete_pixel_runtime import read_pixel_checkpoint,load_pixel_actor
from ilm.visual_lm.native_prompt_interface import prepare_text_prompt,prepare_instruction_image,validate_native_answer
from scripts.infer_native_chinese_v80 import ensure_host_renderer
from scripts.preflight_native_pixel_actor import CODEC,CONTROLS
from scripts.train_conditional_pixel_density import atomic_tensor


def save_answer(result,folder, *, page_width):
    folder.mkdir(exist_ok=False);atomic_tensor(folder/'result.pt',result)
    images=result['images'];files={'result.pt':sha(folder/'result.pt')};ranges=None;page_error=None
    if len(images):
        strip=torch.cat([block[None] for block in images],dim=-1)
        pixels=((strip[0,0]+1)*127.5).to(torch.uint8).numpy()
        Image.fromarray(pixels).save(folder/'answer.png');files['answer.png']=sha(folder/'answer.png')
        try:
            page=fold_writing_prompt(strip,maximum_width=page_width)
        except ValueError as error:
            # Keep the original output if dense ink has no safe visual break.
            page_error=str(error)
        else:
            if not torch.equal(page.unfold(),strip):raise RuntimeError('display wrapping changed the physical stream')
            Image.fromarray(((page.pixels[0,0]+1)*127.5).to(torch.uint8).numpy()).save(folder/'answer_page.png')
            files['answer_page.png']=sha(folder/'answer_page.png');ranges=page.source_ranges
    receipt=dict(blocks=len(images),stopped=result['stopped'],capped=result['capped'],
        files_sha256=files,page_source_ranges=ranges,page_layout_error=page_error,
        empty_answer_no_placeholder=not len(images),output_pixels_repaired=False,
        output_font_renderer=False,output_OCR=False,reference_images_used=False)
    save_json(folder/'output_receipt.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt',type=Path);source.add_argument('--text');source.add_argument('--text-file',type=Path)
    parser.add_argument('--instruction');parser.add_argument('--layout',choices=('page','strip'))
    parser.add_argument('--training-run',type=Path,default=ROOT/'artifacts/complete_pixel_language_v1')
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--maximum-blocks',type=int,choices=range(1,129),default=128,metavar='1..128')
    parser.add_argument('--page-width',type=int,default=1024)
    args=parser.parse_args()
    if args.output.exists():parser.error('preserve the previous answer and choose a fresh output directory')
    if (args.instruction is not None or args.layout is not None) and args.prompt is None:
        parser.error('--instruction and --layout apply only to image input')
    if args.page_width<64 or args.page_width%32:parser.error('--page-width must be a multiple of 32, at least 64')
    if args.text is not None or args.text_file is not None or args.instruction is not None:ensure_host_renderer()
    torch.set_num_threads(4)
    if args.prompt:
        prepared=prepare_image_prompt(args.prompt,layout=args.layout or 'page')
        if args.instruction is not None:prepared=prepare_instruction_image(args.instruction,prepared)
    else:
        text=args.text if args.text is not None else args.text_file.read_text(encoding='utf-8')
        prepared=prepare_text_prompt(text)
    args.output.mkdir(parents=True)
    inputs=save_prepared_prompt(prepared,args.output/'input')
    if args.prepare_only:
        report=dict(input=inputs,preparation_only=True,checkpoint_loaded=False,
            CUDA_initialized=torch.cuda.is_initialized(),generation_performed=False,language_quality_claim=False)
        save_json(args.output/'report.json',report);print(json.dumps(report,indent=2));return
    checkpoint=args.checkpoint or args.training_run/'execution/checkpoint_latest.pt'
    with gpu_zero() as device:
        begun=time.perf_counter()
        state,raster,identity=read_pixel_checkpoint(checkpoint,args.training_run,allow_incomplete=args.allow_incomplete)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        local=ROOT.parent/'LocalLLM/.local'
        actor,provenance=load_pixel_actor(state,raster,controls=CONTROLS,codec=CODEC,
            ollama_root=local/'models/ollama',gguf_source=local/'tools/llama.cpp/gguf-py',device=device)
        loaded=time.perf_counter()
        result=actor.generate(prepared.pixels,maximum_blocks=args.maximum_blocks)
        validate_native_answer(result,args.maximum_blocks)
        output=save_answer(result,args.output/'output',page_width=args.page_width)
        peak=torch.cuda.max_memory_allocated()/1024**3
        if peak>22.:raise RuntimeError('standalone pixel inference exceeds its 22 GiB allocation budget')
        save_json(args.output/'report.json',dict(input=inputs,output=output,checkpoint=identity,source=provenance,
            model_load_seconds=loaded-begun,generation_and_saving_seconds=time.perf_counter()-loaded,
            peak_allocated_vram_gib=peak,actual_prompt_PNG_invoked=True,teacher_service=False,
            tokenizer=False,OCR=False,reference_images_used=False,language_quality_claim=False))
        print(json.dumps(dict(output=str(args.output/'output'),blocks=len(result['images']),stopped=result['stopped'],
            capped=result['capped'],checkpoint_update=identity['checkpoint_update']),indent=2))


if __name__=='__main__':main()
