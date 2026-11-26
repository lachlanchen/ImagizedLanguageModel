"""Measure how field location affects the same trained writer on unchanged pictured inputs."""
from collections import defaultdict
import json
from zipfile import ZipFile
import numpy as np
import torch
from torch.nn import functional as F
from ilm.visual_lm.full_image_roles import pictured_input
from scripts.evaluate_full_image_roles import ARCHIVE
from scripts.prepare_full_image_roles import BASE,PUB,WORK,ROOT,Ink,read,dump,sha
from scripts.train_ordered_translation import pack
from scripts.train_paired_ink import material
from scripts.write_full_image_roles import PicturedTranslator
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


@torch.inference_mode()
def main():
    configure_cuda_cli()
    with gpu_owner():
        with ZipFile(ARCHIVE) as z:
            rows=json.loads(z.read("records.json"))
        assert sha(ARCHIVE)==read(PUB/"generation.json")["archive"]
        writer=PicturedTranslator("marked","cuda")
        feat,inv,local=material("cuda")
        face=Ink(writer.palette)
        metrics=defaultdict(lambda:defaultdict(float))
        for t in (0,1):
            for group in ("book","plain","subject","object"):
                eligible=[r for r in rows if r["t"]==t and r["group"]==group]
                for j in range(0,len(eligible),24):
                    chunk=eligible[j:j+24]
                    src=pack([face.encode(r["source"]) for r in chunk],"cuda")
                    y=pack([face.encode(r["ref"])+[1] for r in chunk],"cuda")
                    assert src.max()<len(feat) and y.max()<len(feat)
                    fields=(255*writer.marker(feat,src).sigmoid()*src.ne(0).unsqueeze(-1)).round().byte()
                    changed=fields.clone()
                    for i,r in enumerate(chunk):
                        n=len(r["source"])
                        changed[i,:n]=fields[i,:n].roll(n//2,0)
                    q=local[t][y]
                    valid=q.ne(-100)
                    bank=writer.writer.optical(feat)[src]*src.ne(0).unsqueeze(-1)
                    win=F.pad(bank,(0,0,0,3)).unfold(1,4,1).transpose(2,3).flatten(2)
                    x=writer.writer.window(win)+writer.writer.pos[:src.size(1)]
                    m=metrics[group]
                    m["source_state_norm"]+=float((x.norm(dim=-1)*src.ne(0)).sum())
                    m["field_norm"]+=float((writer.writer.role_field(fields.float()/255).norm(dim=-1)*src.ne(0)).sum())
                    m["positions"]+=int(valid.sum())
                    m["requests"]+=len(chunk)
                    m["changed_free_images"]+=sum(r["answers"]["marked"]["png"]!=r["answers"]["displaced"]["png"] for r in chunk)
                    baseline=None
                    for label,cues in (("predicted",fields),("displaced",changed),("zero",torch.zeros_like(fields))):
                        logits=writer.writer(feat,pictured_input(src,cues),y,inv[t],t)
                        choice=logits.argmax(-1)
                        if baseline is None:
                            baseline=choice
                        m[label+"_nll"]+=float(F.cross_entropy(logits.transpose(1,2),q,ignore_index=-100,reduction="sum"))
                        m[label+"_correct"]+=int((valid&(choice==q)).sum())
                        m[label+"_changed_decisions"]+=int((valid&(choice!=baseline)).sum())
        for m in metrics.values():
            for mode in ("predicted","displaced","zero"):
                m[mode+"_nll"]/=m["positions"]
            m["relative_field_magnitude"]=m["field_norm"]/m["source_state_norm"]
        dump(PUB/"influence.json",dict(groups=metrics,archive=sha(ARCHIVE),
             protocol=sha(ROOT/"references/full_image_roles_sensitivity_protocol.md")))
        print(dict(metrics["book"]),flush=True)


if __name__=="__main__":
    main()
