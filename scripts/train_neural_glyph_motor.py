"""Fit a shared motor to the frozen literary translator's visual embeddings."""
import math
from time import monotonic
import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F
from ilm.visual_lm.neural_glyph_motor import MotorTranslator
from scripts.write_neural_glyphs import MODEL
from scripts.train_visual_prefix import atomic_save, gpu_owner, material
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_ordered_translation import MODEL as BASE, restore, read, dump, sha


def main():
    configure_cuda_cli()
    with gpu_owner():
        torch.set_num_threads(2)
        torch.manual_seed(801373)
        torch.cuda.reset_peak_memory_stats()
        begin=monotonic()
        f,inventory,_=material("cuda")
        net=MotorTranslator(*(len(v)-1 for v in inventory)).cuda()
        net.writer=restore("role-grounded","cuda").requires_grad_(False)
        union=torch.unique(torch.cat([v[1:] for v in inventory]),sorted=True)
        with torch.no_grad():
            all_codes=net.writer.optical(f)
            for t,v in enumerate(inventory):
                getattr(net,f"codes_{t}").copy_(all_codes[v[1:]])
            codes=all_codes[union]
        gold=torch.from_numpy(np.unpackbits(np.load(BASE/"palette.npy"),axis=1)).cuda()[union].float()
        opt=torch.optim.AdamW(net.motor.parameters(),lr=.001,weight_decay=.001,fused=True)
        for step in range(4000):
            i=torch.randint(len(codes),(256,),device="cuda")
            opt.param_groups[0]["lr"]=.00012+.00054*(1+math.cos(math.pi*step/3999))
            loss=F.binary_cross_entropy_with_logits(net.motor(codes[i]),gold[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            diff=(net.motor(codes)>0)!=gold.bool()
            pixels=int(diff.sum())
            imperfect=int(diff.any(1).sum())
        torch.cuda.synchronize()
        seconds=monotonic()-begin
        peak=torch.cuda.max_memory_allocated()
        if seconds>1800 or peak>12*2**30: raise RuntimeError("Fit ceiling")
        MODEL.mkdir(parents=True,exist_ok=True)
        path=MODEL/"weights.safetensors"
        atomic_save(lambda p:save_file({k:v.detach().cpu() for k,v in net.state_dict().items()},p),path)
        dump(MODEL/"fit.json",dict(weights=sha(path),parent=read(BASE.parent/"library_roles/grounded/fit.json")["weights"],
            choices=[len(v)-1 for v in inventory],distinct=len(union),imperfect=imperfect,incorrect_pixels=pixels,
            updates=4000,seconds=seconds,peak_bytes=peak,seed=801373,parameters=sum(p.numel() for p in net.parameters()),
            motor_parameters=sum(p.numel() for p in net.motor.parameters()),choice_vector_scalars=sum(v.numel() for k,v in net.named_buffers() if k.startswith("codes_"))))
        print("imperfect",imperfect,"of",len(union),"pixels",pixels,flush=True)


if __name__=="__main__":
    main()
