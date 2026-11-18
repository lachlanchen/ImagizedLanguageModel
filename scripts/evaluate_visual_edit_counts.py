"""Compare fixed and learned insertion allowances on the identical frozen editor."""
import fcntl
import time

import torch
from safetensors.torch import load_file

from ilm.visual_lm.visual_edit_fertility import FertilityEditor
from scripts.evaluate_visual_edit_translation import (OUT,ROOT,bank,evaluate,guard,inputs,load,
    local_program,require,sha,write_new)


class FixedCountEditor(FertilityEditor):
    def insertion_counts(self,memory,global_state):
        return torch.full(memory.shape[:2],self.count,dtype=torch.long,device=memory.device)


def main():
    start=time.monotonic()
    guard(start)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    directory=OUT/'fertility'
    fit=load(directory/'report.json')
    path=directory/'editor.safetensors'
    require(sha(path)==fit['weights_sha256'],'Count model changed')
    state=load_file(str(path))
    tiles=bank()
    train=inputs('train')
    program=local_program(train)
    for name,kind in [('fixed',FixedCountEditor),('learned',FertilityEditor)]:
        net=kind(state['landmarks'],fit['maximum']).cuda().eval()
        net.load_state_dict(state)
        if name=='fixed':
            net.count=fit['constant_control']
        folder=directory/name
        folder.mkdir()
        scores=evaluate('heldout',net,tiles,program,train,start,folder/'heldout')
        if name=='learned' and scores['outcome']['translation']['hard_exact']:
            evaluate('confirmation',net,tiles,program,train,start,folder/'confirmation')
    write_new(directory/'evaluation.json',dict(seconds=time.monotonic()-start,
              peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
