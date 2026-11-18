"""Apply a constrained decoder to the frozen varied planner and score complete output images."""
from itertools import product
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_model
from scripts import test_official_passage_composition as external
from scripts.compare_passage_instructions import OUT, TEST, internal
from scripts.train_paired_passage import ROOT,Corpus
from scripts.train_visual_extractive_reader import load_cases
from scripts.prepare_pictured_entry_answers import load,sha,write_json
from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer,legal_plan

DESTINATION=OUT/"ordered_decoder"


def check_paths():
    # Enumerate every path, independent of the tensor recurrence.
    random=np.random.default_rng(309)
    for length in range(1,6):
        logits=random.normal(size=(5,6,4)).astype(np.float32)
        sizes=torch.tensor([length,1,2,3,5])
        result=legal_plan(torch.from_numpy(logits),sizes).numpy()
        for i,n in enumerate(sizes.tolist()):
            best=-np.inf
            for seq in product(range(4),repeat=n):
                active=False
                stopped=False
                prev=0
                for action in seq:
                    if stopped and action!=0 or action==2 and prev!=1 or action==0 and active and not stopped:
                        break
                    if action:active=True
                    if action==3:stopped=True
                    prev=action
                else:
                    best=max(best,sum(logits[i,j,v] for j,v in enumerate(seq)))
            assert abs(sum(logits[i,j,v] for j,v in enumerate(result[i,:n]))-best)<1e-5


def main():
    for path,digest in load(ROOT/"references/evidence/passage_ordered_decoder_registration.json")["files"].items():
        assert sha(ROOT/path)==digest
    torch.set_num_threads(2)
    check_paths()
    DESTINATION.mkdir(exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(.86)
    began=monotonic()
    net=OrderedPassageComposer().eval()
    load_model(net,str(OUT/"varied.safetensors"))
    net.cuda()
    external.WRAPPERS=TEST
    external.OUT=DESTINATION
    external_set=external.prepare()
    official=external.evaluate_checkpoint(net,"ordered",external_set)
    _,dev=load_cases(began)
    data=Corpus("development",dev)
    images,stops=internal(net,data)
    n=len(data.pairs)
    file=DESTINATION/"internal.npz"
    np.savez_compressed(file,pixels=np.frombuffer(b"".join(images),np.uint8).reshape(-1,128),
                        offsets=np.cumsum([0]+[len(im)//128 for im in images]),stopped=stops)
    with np.load(file) as z:
        pix,off,stops=z["pixels"],z["offsets"],z["stopped"]
    ok=np.array([bool(stops[i]) and pix[a:b].tobytes()==data.target(i) for i,(a,b) in enumerate(zip(off[:-1],off[1:],strict=True))])
    result=dict(official=official,internal_pairs=int(ok[:n].sum()),internal_both=int(ok[:n].reshape(-1,2).all(1).sum()),
                single=int(ok[n:].sum()),internal_archive_sha256=sha(file),seconds=monotonic()-began)
    write_json(DESTINATION/"result.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
