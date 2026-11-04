#!/usr/bin/env python3
"""Read a native prompt image and freely generate whole-image answer blocks."""

import argparse
import json
from pathlib import Path
import shutil

import torch

from ilm.visual_lm.correlated_native_runtime import generate_correlated, load_correlated
from ilm.visual_lm.visual_cell_data import pack_visual_cells
from ilm.visual_lm.visual_qa_data import VisualQARenderer
from ilm.visual_lm.visual_qa_evaluation import save_answer_image
from scripts.infer_exact_raster_density_v54 import load_aligned_image
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _atomic_save
from scripts.train_correlated_native_density_v62 import configure_gpu
from scripts.train_native_pixel_instruction_v61 import DEFAULT_SNAPSHOT


def export_bundle(checkpoint,snapshot,folder):
    folder=Path(folder)
    if folder.exists():raise FileExistsError("deployment bundle already exists")
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    folder.mkdir(parents=True);architecture=folder/"architecture";architecture.mkdir()
    for name,digest in payload["external_code_sha256"].items():
        source=Path(snapshot)/name
        if sha256(source)!=digest:raise ValueError("external architecture code differs")
        shutil.copy2(source,architecture/name)
    actor=folder/"actor";actor.mkdir();(actor/"__init__.py").write_text("")
    for name in ("correlated_native_density.py","correlated_native_runtime.py","native_pixel_instruction.py"):
        shutil.copy2(Path("ilm/visual_lm")/name,actor/name)
    shutil.copy2(checkpoint,folder/"weights.pt")
    shutil.copy2("scripts/run_correlated_native_bundle.py",folder/"run.py")
    shutil.copy2("/usr/share/common-licenses/Apache-2.0",folder/"LICENSE-foundation")
    manifest=dict(source_checkpoint_sha256=sha256(checkpoint),components=payload["components"],
        external_foundation="MIXAR-477M; TYTTYTTYT/llama-pixar-477m-p32-stage2; Apache-2.0",
        external_pretraining_is_not_local_training=True,adapted_on="CMRC2018, CC-BY-SA-4.0",
        files_sha256={str(p.relative_to(folder)):sha256(p) for p in folder.rglob("*") if p.is_file()})
    _atomic_json(manifest,folder/"deployment_manifest.json")


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--snapshot",default=str(DEFAULT_SNAPSHOT))
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-image");source.add_argument("--prompt-text")
    parser.add_argument("--out",required=True);parser.add_argument("--export")
    args=parser.parse_args();device=configure_gpu()
    model=load_correlated(args.checkpoint,args.snapshot,device)
    prompt=((load_aligned_image(args.prompt_image)>=.5).float() if args.prompt_image
            else VisualQARenderer()(args.prompt_text))
    result=generate_correlated(model,prompt.to(device));images=result.pop("images").cpu()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    save_answer_image(images,out);_atomic_save(dict(images=images),out.with_suffix(".pt"))
    pack_visual_cells(prompt,columns=len(prompt),gutter=0).save(out.with_name(out.stem+"_prompt.png"))
    result.update(prompt_cells=len(prompt),generated_cells=len(images),tokenizer=False,
        unicode_ids=False,ocr=False,glyph_inventory=False,external_model_call=False,
        output_placeholder_only=not len(images),qualification="research model; see locked audit")
    _atomic_json(result,out.with_suffix(".json"))
    if args.export:export_bundle(args.checkpoint,args.snapshot,args.export)
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
