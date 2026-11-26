"""Measure paired end-to-end inference and independently check every emitted image."""
from hashlib import sha256
from io import BytesIO
import json
from statistics import median
from time import perf_counter
from zipfile import ZipFile
from pathlib import Path

import numpy as np
import torch
from ilm.visual_lm.compiled_glyph_stream import CompiledGlyphStream
from ilm.visual_lm.target_glyph_teaching import load_writer
from scripts.evaluate_print_translation import PUB, ROOT, ARCHIVE, dump, sha, read
from scripts.train_field_to_glyph import configure_cuda_cli, gpu_owner
from scripts.write_neural_glyphs import grid_bits, save_ink


def clock():
    torch.cuda.synchronize()
    return perf_counter()


def inputs():
    assert sha(ARCHIVE)==read(PUB/"images.json")["archive"]
    batches=[]
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read("outcomes.json"))
        for t,lang in enumerate(("en","zh")):
            all_i=[i for i,r in enumerate(rows) if r["to"]==lang]
            for j in range(0,len(all_i),24):
                ids=all_i[j:j+24]
                lengths=[len(rows[i]["source"]) for i in ids]
                images=np.zeros((len(ids),max(lengths),32,32),dtype=bool)
                for k,(i,n) in enumerate(zip(ids,lengths,strict=True)):
                    images[k,:n]=grid_bits(BytesIO(z.read(f"{i:04d}/canonical.png")),n)
                batches.append((ids,torch.from_numpy(images).cuda(),torch.tensor(lengths,device="cuda"),t))
    return rows,batches


def traversal(actor,batches):
    return [actor.write(img,lens,t) for _,img,lens,t in batches]


def hashes(parts,batches):
    result={}
    for (ids,_,_,_), (images,counts,stops) in zip(batches,parts,strict=True):
        for i,grid,n,end in zip(ids,images.cpu().numpy(),counts.tolist(),stops.tolist(),strict=True):
            out=BytesIO()
            save_ink(grid[:n],out,image_format="PNG")
            result[i]=(sha256(out.getvalue()).hexdigest(),n,end)
    return result


def main():
    configure_cuda_cli()
    with gpu_owner():
        torch.set_num_threads(2)
        rows,batches=inputs()
        model=load_writer("guided","cuda")
        start=clock()
        fast=CompiledGlyphStream(model)
        compile_seconds=clock()-start
        tables=sum(v.numel()*v.element_size() for tb in fast.tables.values() for v in (tb.pixels,tb.feedback,tb.keys))
        actors=dict(direct=model,compiled=fast)
        original={i:(r["answers"]["guided"]["png"],r["answers"]["guided"]["cells"],r["answers"]["guided"]["stop"]) for i,r in enumerate(rows)}
        mismatches={}
        for name,actor in actors.items():
            output=hashes(traversal(actor,batches),batches)
            mismatches[name]=[dict(i=i,result=output[i],archived=a) for i,a in original.items() if output[i]!=a]
        samples=[]
        for k in range(5):
            tt={}
            for name in (("direct","compiled") if k%2==0 else ("compiled","direct")):
                start=clock()
                emitted=traversal(actors[name],batches)
                tt[name]=clock()-start
                del emitted
            samples.append(tt)
        speed=median(v["direct"]/v["compiled"] for v in samples)
        stats=dict(archive=sha(ARCHIVE),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),n=len(rows),
                   samples=samples,compile_seconds=compile_seconds,table_bytes=tables,speedup=speed,
                   archived_mismatches=mismatches,passed=not any(mismatches.values()) and speed>1,
                   median_cold_seconds=median(v["compiled"] for v in samples)+compile_seconds,
                   program=sha(Path(__file__)),implementation=sha(ROOT/"ilm/visual_lm/compiled_glyph_stream.py"),
                   protocol=sha(ROOT/"references/glyph_stream_cost_protocol.md"))
        dump(PUB/"inference.json",stats)
        print("speed",speed,"mismatches",{k:len(v) for k,v in mismatches.items()},flush=True)


if __name__=="__main__":
    main()
