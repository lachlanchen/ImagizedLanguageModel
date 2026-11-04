#!/usr/bin/env python3
"""Native PNG or host-rendered text to freely generated answer pixels."""

import argparse
import json
from pathlib import Path
import shutil

import torch

from ilm.visual_lm.native_pixel_instruction import generate
from ilm.visual_lm.native_pixel_runtime import load_trained
from ilm.visual_lm.visual_cell_data import pack_visual_cells
from ilm.visual_lm.visual_qa_data import VisualQARenderer
from ilm.visual_lm.visual_qa_evaluation import save_answer_image
from scripts.infer_exact_raster_density_v54 import load_aligned_image
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _atomic_save
from scripts.train_native_pixel_instruction_v61 import DEFAULT_SNAPSHOT


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-image"); source.add_argument("--prompt-text")
    p.add_argument("--out", required=True)
    p.add_argument("--export", help="self-contained learned weights, config and author code directory")
    a = p.parse_args()
    torch.set_num_threads(4); torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    model = load_trained(a.checkpoint, a.snapshot, device)
    prompt = ((load_aligned_image(a.prompt_image) >= .5).float() if a.prompt_image
              else VisualQARenderer()(a.prompt_text))
    if not 1 <= len(prompt) <= 480:
        raise ValueError("native prompt must have 1–480 aligned image cells")
    result = generate(model, prompt.to(device))
    images = result.pop("images").cpu()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    save_answer_image(images, out)
    _atomic_save(dict(images=images), out.with_suffix(".pt"))
    pack_visual_cells(prompt, columns=len(prompt), gutter=0).save(out.with_name(out.stem+"_prompt.png"))
    result.update(prompt_cells=len(prompt), generated_cells=len(images),
        image_only=True, glyph_inventory=False, tokenizer=False, ocr=False, external_model_call=False,
        probability="native causal block Bernoulli pixels; answer-only adapted",
        qualification="research model; consult completed audit", output_placeholder_only=not len(images))
    _atomic_json(result, out.with_suffix(".json"))
    if a.export:
        folder = Path(a.export)
        if folder.exists():
            raise FileExistsError("refuse to replace an existing deployment bundle")
        model.save_pretrained(folder, safe_serialization=True)
        license_path = Path("/usr/share/common-licenses/Apache-2.0")
        if license_path.exists():
            shutil.copy2(license_path, folder/"LICENSE")
        receipt = dict(external_foundation="MIXAR-477M, TYTTYTTYT/llama-pixar-477m-p32-stage2",
            external_pretraining_is_not_local_training=True, adapted_on="CMRC2018, CC-BY-SA-4.0",
            source_checkpoint_sha256=sha256(a.checkpoint),
            files_sha256={str(f.relative_to(folder)): sha256(f) for f in folder.rglob("*") if f.is_file()})
        _atomic_json(receipt, folder/"deployment_manifest.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
