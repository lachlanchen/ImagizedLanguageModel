"""Measure complete image-to-image writing time at matched single and batched requests."""
from statistics import median
from time import perf_counter

import numpy as np
import torch

from scripts.evaluate_strip_restoration import panel
from scripts.reserve_visual_strip_restoration import BASE, PUB, ROOT, dump, sha
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_strip_restoration import StripWriter


def main():
    configure_cuda_cli()
    torch.set_num_threads(2)
    with gpu_owner():
        rows = panel("book")
        palette = np.unpackbits(np.load(BASE/"palette.npy"),axis=1).reshape(-1,32,32).astype(bool)
        groups = [[palette[r["input"]] for r in rows if r["t"]==t] for t in (0,1)]
        results = []
        for arm in ("parent","mask","restore"):
            writer = StripWriter(arm,"cuda")
            for batch in (1,16):
                for t,images in enumerate(groups):
                    writer.write(images[:batch],t)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                trials=[]
                for _ in range(3):
                    cells=stops=0
                    start=perf_counter()
                    for t,images in enumerate(groups):
                        for j in range(0,len(images),batch):
                            out=writer.write(images[j:j+batch],t)
                            cells+=sum(len(ink) for ink,_ in out)
                            stops+=sum(stop for _,stop in out)
                    torch.cuda.synchronize()
                    trials.append(dict(seconds=perf_counter()-start,cells=cells,stops=stops))
                results.append(dict(arm=arm,batch=batch,median_seconds=median(r["seconds"] for r in trials),
                                    peak_bytes=torch.cuda.max_memory_allocated(),trials=trials))
                print(arm,batch,results[-1]["median_seconds"],flush=True)
            del writer
        dump(PUB/"latency.json",dict(device=torch.cuda.get_device_name(),requests=len(rows),
            protocol=sha(ROOT/"references/visual_strip_latency_protocol.md"),results=results))


if __name__=="__main__":
    main()
