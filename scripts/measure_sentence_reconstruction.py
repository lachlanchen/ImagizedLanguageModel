"""Measure GPU batching on archived requests, checking every output against its graded image."""
from io import BytesIO
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch
from ilm.visual_lm.compiled_glyph_stream import CompiledGlyphStream
from ilm.visual_lm.sentence_reconstruction import restore
from scripts.check_field_writer_roundtrip import cells
from scripts.evaluate_sentence_reconstruction import ARCHIVE, PUB, configure_cuda_cli, gpu_owner, grid_bits, requests, dump, read, sha
from scripts.measure_span_gpu_cost import BoardMeter


@torch.inference_mode()
def run(model, source, groups, batch):
    outputs=[]
    for t,ids in enumerate(groups):
        for j in range(0,len(ids),batch):
            ix=ids[j:j+batch]
            images=[source[i] for i in ix]
            padded=torch.nn.utils.rnn.pad_sequence(images,batch_first=True).cuda()
            ink,n,stop=model.write(padded,torch.tensor(list(map(len,images)),device="cuda"),t)
            ink=ink.cpu().numpy()
            outputs.extend((i,ink[k,:size],done) for k,(i,size,done) in enumerate(zip(ix,n.tolist(),stop.tolist(),strict=True)))
    return outputs


def main():
    configure_cuda_cli()
    record=PUB/"gpu-cost.json"
    assert sha(ARCHIVE)==read(PUB/"generation.json")["archive"]
    if record.exists():
        assert read(record)["archive"]==sha(ARCHIVE)
        return
    rows=requests()
    with ZipFile(ARCHIVE) as z:
        records=json.loads(z.read("records.json"))
        source=[torch.from_numpy(grid_bits(BytesIO(z.read(f"{i:04d}/source.png")),len(r["input"]))) for i,r in enumerate(rows)]
        refs=[cells(z.read(f"{i:04d}/preference.png"),r["answers"]["preference"]["cells"]) for i,r in enumerate(records)]
    groups=[[i for i,r in enumerate(rows) if r["t"]==t] for t in (0,1)]
    settings=(("live",12),("live",96),("compiled",12),("compiled",48),("compiled",96))
    trials=[]
    with gpu_owner():
        torch.set_num_threads(2)
        meter=BoardMeter()
        try:
            net=restore("preference","cuda")
            models={"live":net,"compiled":CompiledGlyphStream(net)}
            for mode,batch in settings:
                run(models[mode],source,groups,batch)
            for order in (settings,settings[::-1],settings):
                for mode,batch in order:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    e0=meter.joules()
                    start=perf_counter()
                    emitted=run(models[mode],source,groups,batch)
                    torch.cuda.synchronize()
                    sec=perf_counter()-start
                    energy=meter.joules()-e0
                    diffs=[i for i,ink,stop in emitted if stop!=records[i]["answers"]["preference"]["stop"]
                           or not np.array_equal(ink,refs[i])]
                    trials.append(dict(mode=mode,batch=batch,requests=len(emitted),seconds=sec,joules=energy,
                                       peak=torch.cuda.max_memory_allocated(),differences=diffs))
        finally:
            meter.ok(meter.lib.nvmlShutdown())
    summary={}
    for mode,batch in settings:
        sel=[v for v in trials if v["mode"]==mode and v["batch"]==batch]
        summary[f"{mode}-{batch}"]=dict(requests_per_second=median(len(rows)/v["seconds"] for v in sel),
                    joules_per_request=median(v["joules"]/len(rows) for v in sel),identical=all(not v["differences"] for v in sel))
    dump(record,dict(archive=sha(ARCHIVE),hardware=torch.cuda.get_device_name(),trials=trials,summary=summary,program=sha(Path(__file__))))
    print(summary,flush=True)


if __name__=="__main__":
    main()
