"""Score a reproducible unfitted writer on exactly the same open-writing pages."""

import numpy as np
import torch
from ilm.visual_lm.discourse_writer import DiscourseWriter
from scripts.evaluate_visual_discourse import cohorts,lp
from scripts.prepare_visual_discourse import PUBLIC,dump
from scripts.train_visual_discourse import SEED,Corpus
from scripts.train_span_choice import gpu_owner


@torch.inference_mode()
def main():
    data=Corpus();_,groups=cohorts(data)
    with gpu_owner():
        # The training initializer allocates its parameters on CPU before .cuda().
        torch.manual_seed(SEED);model=DiscourseWriter().cuda().eval()
        bank=model.encode(data.feature("ink","cuda"));out={}
        for k,rows in groups.items():
            order=sorted(range(len(rows)),key=lambda i:len(rows[i]["x"]))
            if len(order)%2:order.pop()
            rows=[rows[i] for i in sorted(order)]
            scores,n=lp(model,bank,[(r["x"],r["y"]) for r in rows],data.n)
            out[k]=dict(pages=len(rows),cells=int(n.sum()),bits_per_glyph=float(-scores.sum()/n.sum()/np.log(2)))
        dump(PUBLIC/"initial-open.json",out)


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli();main()
