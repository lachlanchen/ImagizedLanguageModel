#!/usr/bin/env python3
"""Measure native reading and assisted writing separately; never an ILM score."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image
import torch

from ilm.native_unified.showo2_source import load, image_latent, reading_prefix, sha

ANSWER_INSTRUCTION = "Follow the request written in the image. Output only the requested answer, with no explanation."
TRANSCRIBE_INSTRUCTION = "Transcribe all the text in the image, preserving line order. Output only the transcription."
WRITING_TEMPLATE = 'A flat scan of a white page with a single line of black text: "{string}". Plain readable print, generous white margins, no decoration, no other text.'


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def prepare(args):
    original = jsonl(args.manifest)
    strips = {x["id"]: x for x in jsonl(args.copy_manifest)}
    copies = [r for split in ("train", "validation") for r in
              [x for x in original if x["task"] == "copy" and x["split"] == split][:4]]
    qa = [r for r in original if r["task"] == "qa" and r["split"] == "validation"][:4]
    if len(copies) != 8 or len(qa) != 4:
        raise ValueError("declared source cohort unavailable")
    inputs, references, writing = [], [], []
    (args.output / "inputs").mkdir()
    (args.output / "references").mkdir()
    for i, original_row in enumerate(copies + qa):
        is_copy = original_row["task"] == "copy"
        row = strips[original_row["id"]] if is_copy else original_row
        folder = args.copy_manifest.parent if is_copy else args.manifest.parent
        path = folder / row["input"]
        if sha(path) != row["input_sha256"]:
            raise ValueError("input source changed")
        with Image.open(path) as im:
            if im.size != ((1024, 256) if is_copy else (1024, 1024)):
                raise ValueError("input source geometry differs")
        name = f"inputs/{i:02}.png"; shutil.copy2(path, args.output / name)
        target = args.manifest.parent / original_row["target"]
        if sha(target) != original_row["target_sha256"]:
            raise ValueError("reference source changed")
        reference = f"references/{i:02}.png"; shutil.copy2(target, args.output / reference)
        inputs.append(dict(id=row["id"], input=name, input_sha256=sha(path), task=row["task"], seed=20268521+i))
        references.append(dict(id=row["id"], task=row["task"], language=row["language"],
            expected=original_row["reference_text"], target=reference, target_sha256=sha(target),
            split=row["split"]))
        if is_copy:
            writing.append(dict(id=row["id"], string=original_row["reference_text"],
                prompt=WRITING_TEMPLATE.format(string=original_row["reference_text"]), seed=20268721+i))
    save(args.output / "inputs.json", inputs)
    save(args.output / "references.json", references)
    save(args.output / "assisted_writing.json", writing)
    save(args.output / "preparation.json", dict(manifest_sha256=sha(args.manifest),
        copy_manifest_sha256=sha(args.copy_manifest), inputs_sha256=sha(args.output/"inputs.json"),
        references_sha256=sha(args.output/"references.json"), writing_sha256=sha(args.output/"assisted_writing.json"),
        cases=12, reading_calls=20, assisted_writing_calls=8, source_pixels_resized=False,
        expected_answers_in_reading_index=False, optimizer_updates=0, student_language_test=False))


@contextmanager
def gpu_slot():
    lines = {s.split(":")[0]: int(s.split()[1]) for s in Path("/proc/meminfo").read_text().splitlines() if s.split()[1].isdigit()}
    if lines["MemAvailable"] < 24 * 1024**2 or (lines["SwapTotal"] and 1-lines["SwapFree"]/lines["SwapTotal"] > .75):
        raise RuntimeError("shared workstation memory limit")
    Path("artifacts").mkdir(exist_ok=True)
    with Path("artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def save_pixels(folder, pixels):
    folder.mkdir(parents=True, exist_ok=True)
    # Native VAE already postprocesses to [-1,1]. Preserve that complete tensor.
    value = pixels.detach().cpu()
    torch.save(value, folder / "pixels.pt")
    if value.ndim == 5:
        if value.shape[2] != 1:
            raise ValueError("image diagnostic unexpectedly returned multiple frames")
        value = value.squeeze(2)
    if value.shape[0] != 1 or value.shape[1] != 3:
        raise ValueError("one RGB image required")
    array = ((value[0].float().permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(array).save(folder / "image.png")
    return dict(png_sha256=sha(folder/"image.png"), pixels_sha256=sha(folder/"pixels.pt"),
        tensor_shape=list(pixels.shape), output_repair=False)


@torch.inference_mode()
def reading(args, bundle):
    rows = json.loads(args.inputs.read_text())
    if len(rows) != 12:
        raise ValueError("complete fixed twelve-input index required")
    outputs = []
    for i, row in enumerate(rows):
        source = args.inputs.parent / row["input"]
        if sha(source) != row["input_sha256"]:
            raise ValueError("input PNG changed")
        folder = args.output / f"{i:02}"; folder.mkdir()
        z, pixels = image_latent(bundle, source, row["seed"])
        torch.save(z.cpu(), folder / "latent.pt")
        codec = save_pixels(folder / "input_codec", bundle.vae.batch_decode(z.unsqueeze(2)))
        for kind, words, cap in (("answer", ANSWER_INSTRUCTION, 96), ("transcription", TRANSCRIBE_INSTRUCTION, 128)):
            if kind == "transcription" and row["task"] != "copy":
                continue
            embeddings, mask = reading_prefix(bundle, z, words)
            torch.manual_seed(20268621+i); torch.cuda.manual_seed_all(20268621+i)
            started = time.monotonic()
            ids = bundle.model.mmu_generate(input_embeds=embeddings, attention_mask=mask,
                max_new_tokens=cap, temperature=1.0, top_k=1, eos_token=bundle.tokenizer.eos_token_id)
            ids = [int(x) for x in ids]
            entry = dict(id=row["id"], task=row["task"], kind=kind, input_sha256=row["input_sha256"],
                latent_sha256=sha(folder/"latent.pt"), latent_seed=row["seed"], readout_seed=20268621+i,
                native_source_helper=True, student_answer=False, text=bundle.tokenizer.decode(ids,skip_special_tokens=True),
                ids=ids, stopped=bool(ids and ids[-1]==bundle.tokenizer.eos_token_id), decisions=len(ids),
                cap=cap, instruction=words, elapsed_seconds=time.monotonic()-started, codec=codec,
                prefix_shape=list(embeddings.shape),
                prefix_sha256=hashlib.sha256(embeddings.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
            outputs.append(entry);save(folder/(kind+".json"),entry)
            save(args.output/"outputs.json",outputs)
            print(row["id"],kind,repr(entry["text"]),flush=True)
        del z, pixels
    return dict(cases=12, complete_calls=len(outputs), expected_calls=20, input_only_index_sha256=sha(args.inputs))


@torch.inference_mode()
def writing(args, bundle):
    rows = json.loads(args.inputs.read_text())
    if len(rows) != 8:
        raise ValueError("eight fixed explicitly supplied strings required")
    model, native, ids, tok = bundle.model, bundle.native, bundle.ids, bundle.tokenizer
    model.image_position_ids = torch.arange(1024, device=bundle.device)[None]
    tokens_per_image, length = 1025, 1280
    transport = native.transport.create_transport(path_type="Linear", prediction="velocity", loss_weight=None,
        train_eps=None, sample_eps=None, snr_type="lognorm", do_shift=True, seq_len=tokens_per_image)
    sampler = native.transport.Sampler(transport)
    sample = sampler.sample_ode(sampling_method="euler", num_steps=50, atol=1e-6, rtol=1e-3,
        reverse=False, time_shifting_factor=3.0)
    results = []
    for i, row in enumerate(rows):
        if row["prompt"] != WRITING_TEMPLATE.format(string=row["string"]) or row["seed"] != 20268721+i:
            raise ValueError("declared writing condition differs")
        torch.manual_seed(row["seed"]);torch.cuda.manual_seed_all(row["seed"])
        a, b, pa, pb = native.misc.prepare_gen_input([row["prompt"]], tok, tokens_per_image,
            ids["bos_id"], ids["eos_id"], ids["boi_id"], ids["eoi_id"], tok.pad_token_id,
            ids["img_pad_id"], length-tokens_per_image-4, bundle.device)
        text, positions = torch.cat([a,b]), torch.cat([pa,pb])
        if text.shape != (2,length):
            raise ValueError("native generation layout changed")
        mask = native.masks.omni_attn_mask_naive(2,length,positions,bundle.device).to(bundle.dtype)
        z = torch.randn((1,16,64,64)).to(bundle.dtype).to(bundle.device)
        z = torch.cat([z,z])
        started = time.monotonic()
        latent = sample(z, model.t2i_generate, text_tokens=text, attention_mask=mask,
            modality_positions=positions, output_hidden_states=True, max_seq_len=length, guidance_scale=5.0)[-1]
        latent = latent.chunk(2)[0]
        folder = args.output/f"{i:02}";folder.mkdir()
        torch.save(latent.cpu(),folder/"latent.pt")
        raw = bundle.vae.batch_decode(latent.unsqueeze(2))
        entry = dict(id=row["id"], seed=row["seed"], prompt=row["prompt"], expected=row["string"],
            assisted_string_supplied=True, student_answer=False, elapsed_seconds=time.monotonic()-started,
            latent_sha256=sha(folder/"latent.pt"), image=save_pixels(folder,raw), path=f"{i:02}/image.png")
        results.append(entry);save(folder/"result.json",entry);save(args.output/"outputs.json",results)
        print(row["id"],"assisted writing complete",flush=True)
    return dict(complete_calls=len(results),expected_calls=8,assisted_index_sha256=sha(args.inputs),
        time_points=50,ode="native Euler",guidance=5.0,shift_factor=3.0,canvas=[512,512])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("prepare","reading","writing"),required=True)
    p.add_argument("--assets",type=Path,default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--manifest",type=Path,default=Path("artifacts/image_edit_learning/pages_v2/pairs.jsonl"))
    p.add_argument("--copy-manifest",type=Path,default=Path("artifacts/image_edit_learning/strip_pages/pairs.jsonl"))
    p.add_argument("--inputs",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/"status.json",dict(stage=args.stage,state="running"))
    try:
        if args.stage=="prepare":
            prepare(args)
        else:
            if args.inputs is None:raise ValueError("explicit source input index required")
            torch.set_num_threads(8)
            start=time.monotonic()
            with gpu_slot():
                bundle=load(args.assets,dtype=torch.float32 if args.stage=="reading" else torch.bfloat16)
                torch.cuda.reset_peak_memory_stats()
                details=reading(args,bundle) if args.stage=="reading" else writing(args,bundle)
                torch.cuda.synchronize()
                save(args.output/"report.json",dict(stage=args.stage,counts=bundle.counts,details=details,
                    elapsed_seconds=time.monotonic()-start,peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                    implementation_sha256=sha(__file__),source_adapter_sha256=sha(Path(__file__).parents[1]/"ilm/native_unified/showo2_source.py"),
                    optimizer_updates=0,student_language_test=False,language_quality_qualified=False,general_ilm_goal_complete=False))
        save(args.output/"status.json",dict(stage=args.stage,state="completed"))
    except Exception as error:
        save(args.output/"status.json",dict(stage=args.stage,state="failed",error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=="__main__":main()
