"""Measure two image-to-image writers on the same 192 source strips."""
from io import BytesIO
import json
from statistics import median
from time import perf_counter
from zipfile import ZipFile
import numpy as np
import torch
from scripts.measure_span_gpu_cost import BoardMeter
from scripts.check_field_writer_roundtrip import cells
from scripts.evaluate_pictured_path_translation import (ARCHIVE, PUB, ROOT, OpticalSpanWriter, PicturedPathWriter,
    configure_cuda_cli, gpu_owner, grid_bits, requests, dump, read, sha)


def main():
    configure_cuda_cli()
    record=PUB/"gpu-cost.json"
    if record.exists():
        assert read(record)["archive"] == sha(ARCHIVE)
        return
    assert sha(ARCHIVE) == read(PUB/"generation.json")["archive"]
    rows=requests()
    modes={"field":OpticalSpanWriter,"path":PicturedPathWriter}
    with ZipFile(ARCHIVE) as z:
        records=json.loads(z.read("records.json"))
        source=[grid_bits(BytesIO(z.read(f"{i:04d}/source.png")),len(r["input"])) for i,r in enumerate(rows)]
        reference={k:[cells(z.read(f"{i:04d}/{k}.png"),r["answers"][k]["cells"]) for i,r in enumerate(records)] for k in modes}
    trials=[]
    with gpu_owner():
        meter=BoardMeter()
        try:
            models={k:[cls(lang,"cuda") for lang in ("en","zh")] for k,cls in modes.items()}
            groups=[[i for i,r in enumerate(rows) if r["t"]==t] for t in (0,1)]
            def run(k):
                out=[None]*len(rows)
                for model,ids in zip(models[k],groups,strict=True):
                    for i,item in zip(ids,model.write([source[i] for i in ids]),strict=True): out[i]=item
                return out
            for k in modes: run(k)
            for k in ("field","path","path","field","field","path"):
                torch.cuda.synchronize()
                start_energy=meter.joules()
                start=perf_counter()
                output=run(k)
                torch.cuda.synchronize()
                sec=perf_counter()-start
                energy=meter.joules()-start_energy
                changed=[]
                for i,(ink,stop,edges) in enumerate(output):
                    r=records[i]["answers"][k]
                    if stop!=r["stop"] or edges!=r["boundaries"] or not np.array_equal(ink,reference[k][i]): changed.append(i)
                trials.append(dict(mode=k,seconds=sec,joules=energy,differences=changed))
        finally:
            meter.lib.nvmlShutdown()
    scores={k:dict(requests_per_second=median(len(rows)/t["seconds"] for t in trials if t["mode"]==k),
                   board_joules_per_request=median(t["joules"]/len(rows) for t in trials if t["mode"]==k),
                   all_identical=all(not t["differences"] for t in trials if t["mode"]==k)) for k in modes}
    dump(record,dict(archive=sha(ARCHIVE),hardware=torch.cuda.get_device_name(),trials=trials,summary=scores,
                     protocol=sha(ROOT/"references/pictured_path_transfer_protocol.md")))
    print(scores,flush=True)


if __name__=="__main__":
    main()
