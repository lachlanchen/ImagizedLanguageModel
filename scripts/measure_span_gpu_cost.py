"""Measure steady-state complete visual replies and GPU-board energy against frozen outputs."""
from ctypes import CDLL, byref, c_void_p, c_ulonglong
from io import BytesIO
from pathlib import Path
from statistics import median
from time import perf_counter
from zipfile import ZipFile
import json
import numpy as np
import torch
from scripts.evaluate_optical_span_rollout import (ARCHIVE,PUB,ROOT,OpticalSpanWriter,cells,configure_cuda_cli,
    dump,gpu_owner,grid_bits,read,requests,sha)


class BoardMeter:
    def __init__(self):
        self.lib=CDLL("libnvidia-ml.so.1")
        self.handle=c_void_p()
        self.ok(self.lib.nvmlInit_v2())
        self.ok(self.lib.nvmlDeviceGetHandleByIndex_v2(0,byref(self.handle)))

    @staticmethod
    def ok(code):
        if code: raise RuntimeError(f"NVML returned {code}")

    def joules(self):
        current=c_ulonglong()
        self.ok(self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle,byref(current)))
        return current.value/1000


def main():
    configure_cuda_cli()
    path=PUB/"gpu-cost.json"
    assert sha(ARCHIVE)==read(PUB/"generation.json")["archive"]
    if path.exists():
        assert read(path)["archive"]==sha(ARCHIVE)
        return
    rows=requests()
    with ZipFile(ARCHIVE) as z:
        records=json.loads(z.read("records.json"))
        sources=[grid_bits(BytesIO(z.read(f"{i:04d}/source.png")),len(r["input"])) for i,r in enumerate(rows)]
        refs=[cells(z.read(f"{i:04d}/span.png"),r["answers"]["span"]["cells"]) for i,r in enumerate(records)]
    with gpu_owner():
        meter=BoardMeter()
        try:
            models=[OpticalSpanWriter(lang,"cuda") for lang in ("en","zh")]
            indices=[[i for i,r in enumerate(rows) if r["t"]==t] for t in (0,1)]
            def run(size):
                results=[None]*len(sources)
                for model,ix in zip(models,indices,strict=True):
                    for j in range(0,len(ix),size):
                        group=ix[j:j+size]
                        written=model.write([sources[i] for i in group])
                        for i,entry in zip(group,written,strict=True): results[i]=entry
                return results
            run(12)
            trials=[]
            for sizes in ((1,12,48,96),(96,48,12,1),(12,96,1,48)):
                for size in sizes:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    before=meter.joules()
                    tic=perf_counter()
                    outputs=run(size)
                    torch.cuda.synchronize()
                    seconds=perf_counter()-tic
                    energy=meter.joules()-before
                    assert energy>0 and seconds>0
                    diff=[]
                    for i,(ink,stop,edges) in enumerate(outputs):
                        a=records[i]["answers"]["span"]
                        if stop!=a["stop"] or edges!=a["boundaries"] or not np.array_equal(ink,refs[i]): diff.append(i)
                    trial=dict(size=size,seconds=seconds,joules=energy,peak_bytes=torch.cuda.max_memory_allocated(),differences=diff)
                    trials.append(trial)
                    print(size,round(seconds,2),round(energy,1),len(diff),flush=True)
            summary={}
            for size in sizes:
                group=[v for v in trials if v["size"]==size]
                summary[str(size)]=dict(requests_per_second=median(len(rows)/v["seconds"] for v in group),
                    board_joules_per_request=median(v["joules"]/len(rows) for v in group),
                    exact=all(not v["differences"] for v in group))
            dump(path,dict(archive=sha(ARCHIVE),hardware=torch.cuda.get_device_name(),cases=len(rows),trials=trials,
                           summary=summary,program=sha(Path(__file__)),protocol=sha(ROOT/"references/span_gpu_cost_protocol.md")))
        finally:
            meter.lib.nvmlShutdown()


if __name__=="__main__":
    main()
