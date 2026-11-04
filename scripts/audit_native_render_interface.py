#!/usr/bin/env python3
"""CPU-only full-source font and canvas audit of the released pixel renderer."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    revision="c5d5bcfbad2f290296fd83810558f396c6a6333a"
    parser.add_argument("--snapshot",default=str(Path.home()/".cache/huggingface/hub"/
        "models--TYTTYTTYT--llama-pixar-477m-p32-stage2/snapshots"/revision))
    parser.add_argument("--out",default="artifacts/native_render_interface_audit_20261105")
    args=parser.parse_args();out=Path(args.out)
    if (out/"report.json").exists():raise FileExistsError("completed renderer audit is immutable")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("explicit CUDA_VISIBLE_DEVICES='' required")
    if sys.version_info[:2]!=(3,12):raise RuntimeError("use the verified native Python 3.12 environment")
    if "libglib-2.0.so" not in os.environ.get("LD_PRELOAD",""):
        raise RuntimeError("use the documented per-process system GLib/GObject/GIO preloads")
    sys.path.append("/usr/lib/python3/dist-packages")
    # Guard imports before pixar_render; its missing-dependency installer must
    # never run as a side effect of importing the rendering package.
    import cairo
    import gi
    import manimpango
    gi.__path__.append(str(Path("artifacts/external/native_bindings/usr/lib/python3/dist-packages/gi").resolve()))
    gi.require_foreign("cairo")
    gi.require_version("Pango","1.0");gi.require_version("PangoCairo","1.0")
    from gi.repository import Pango,PangoCairo
    import numpy as np
    from PIL import Image
    import torch
    from pixar_render import PixarProcessor
    import pixar_render.processor as processor_module
    import pixar_render.pangocairo_render as backend_module
    from ilm.visual_lm.visual_qa_data import VisualQARenderer
    from ilm.visual_lm.visual_qa_evaluation import save_answer_image

    torch.set_num_threads(2)
    snapshot=Path(args.snapshot);conf=snapshot/"render_config"
    if snapshot.name!=revision:raise ValueError("pinned renderer revision differs")
    processor=PixarProcessor.load(str(conf));backend=processor.renderer
    if processor._block_width!=64 or backend.pixels_per_patch!=32:raise ValueError("native physical layout differs")
    canonical=VisualQARenderer();out.mkdir(parents=True,exist_ok=True)
    paths={"cmrc":"data/visual_qa/cmrc2018_short/train.jsonl",
           "dolly":"data/general_instructions/dolly15k/train.jsonl"}
    expected={"cmrc":3211,"dolly":12070};summaries={};first=[]
    begun=time.perf_counter()
    # A small Cairo surface is sufficient for uncropped Pango measurement;
    # no actual rendering or clipping occurs during layout calculation.
    surface=cairo.ImageSurface(cairo.FORMAT_A8,32,32);context=cairo.Context(surface)

    def measure(text):
        if "\n" in text or "\r" in text:raise ValueError("audit expects the explicit flat rendering view")
        pango_context=PangoCairo.create_context(context)
        pango_context.set_font_description(backend.font)
        layout=Pango.Layout(pango_context);layout.set_text(text,-1)
        width,height=layout.get_pixel_size();ink,_=layout.get_pixel_extents()
        vertical_offset=16-height/2-2
        required_blocks=math.ceil((width+4)/64)
        return dict(logical_width_px=width,logical_height_px=height,unknown_glyphs=layout.get_unknown_glyphs_count(),
            ink_top_px=ink.y+vertical_offset,ink_bottom_px=ink.y+ink.height+vertical_offset,
            vertical_ink_outside=ink.y+vertical_offset<0 or ink.y+ink.height+vertical_offset>32,
            left_ink_outside=2+ink.x<0,
            # One full native block is reserved for the library separator.
            horizontal_overflow=required_blocks>backend.max_pixels_len//64-1,
            required_native_blocks=required_blocks)

    for source,path in paths.items():
        records=[json.loads(s) for s in Path(path).read_text().splitlines()]
        if len(records)!=expected[source]:raise ValueError("training source count changed")
        counts=Counter();widths=[];ratios=[]
        evidence=out/(source+"_layout_measurements.jsonl")
        with evidence.open("w") as writer:
            for index,record in enumerate(records):
                texts={"prompt":record["prompt_flat"] if source=="dolly" else record["prompt"],
                       "answer":record["answer_flat"] if source=="dolly" else record["references"][0]["text"]}
                measurements={key:measure(text) for key,text in texts.items()}
                writer.write(json.dumps(dict(id=record["id"],fields=measurements),ensure_ascii=False)+"\n")
                counts["records"]+=1
                for key,m in measurements.items():
                    counts[key+"_unknown_glyph_records"]+=m["unknown_glyphs"]>0
                    counts[key+"_vertical_clipping_records"]+=m["vertical_ink_outside"]
                    counts[key+"_horizontal_overflow_records"]+=m["horizontal_overflow"]
                    counts[key+"_left_clipping_records"]+=m["left_ink_outside"]
                    counts[key+"_unknown_glyphs"]+=m["unknown_glyphs"]
                    widths.append(m["logical_width_px"])
                    ratios.append(m["required_native_blocks"]/(math.ceil(len(texts[key])/2) or 1))
                safe=all(not any(m[k] for k in ("unknown_glyphs","vertical_ink_outside","horizontal_overflow","left_ink_outside")) for m in measurements.values())
                counts["both_fields_safe"]+=safe
                if index<4:
                    for key,text in texts.items():
                        m=measurements[key];sample=dict(source=source,id=record["id"],field=key,measurement=m)
                        if not m["horizontal_overflow"]:
                            enc=processor.render(text,padding_side="right",add_eos=False,truncate=True,channels=1)
                            pixels=enc.pixel_values[0,0].numpy()
                            Image.fromarray(pixels,mode="L").save(out/f"{source}_{index}_{key}_native.png")
                            try:
                                save_answer_image(canonical(text),out/f"{source}_{index}_{key}_canonical.png")
                                sample["canonical_rendered"]=True
                            except ValueError as error:
                                # Coverage failures are audit evidence, not a
                                # reason to truncate or substitute source text.
                                sample["canonical_rendered"]=False
                                sample["canonical_error"]=str(error)
                            sample["native_width_px"]=pixels.shape[1]
                        first.append(sample)
                if index%1000==0:print(json.dumps(dict(source=source,completed=index+1)),flush=True)
        summaries[source]=dict(counts=counts,median_logical_width_px=float(np.median(widths)),
            median_native_blocks_over_canonical_blocks=float(np.median(ratios)),
            measurement_sha256=digest(evidence),source_sha256=digest(path))
    report=dict(counts=summaries,registered_font_families=sorted(backend.fonts_list),
        primary_font_description=backend.font.to_string(),
        selected_font_runs_inspected=False,first_fixed_samples=first,
        configuration=json.loads((conf/"pixar_processor_conf.json").read_text()),
        renderer_revision=revision,config_sha256=digest(conf/"pixar_processor_conf.json"),
        font_sha256={str(p.relative_to(conf)):digest(p) for p in sorted(conf.rglob("*.ttf"))},
        package_source_sha256={Path(m.__file__).name:digest(m.__file__) for m in (processor_module,backend_module)},
        audit_source_sha256=digest(__file__),protocol_sha256=digest("references/native_render_interface_audit_plan.md"),
        software=dict(torch=torch.__version__,pixar_render=importlib.metadata.version("pixar-render"),pango=Pango.version_string()),
        elapsed_seconds=time.perf_counter()-begun,model_calls=0,gpu_allocations=0,training_data_only=True,
        language_quality_claim=False,scope="full Pango-layout rendering audit, not model evaluation or proof of font-caused language failure")
    (out/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=="__main__":main()
