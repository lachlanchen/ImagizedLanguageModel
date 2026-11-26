"""Read row choices once, then apply the learned optical order to the same values."""
import numpy as np
import torch
from scripts.train_origin_value_reader import layout
from scripts.write_origin_values import strip


@torch.inference_mode()
def predict_two(reader,controller,inputs,device):
    pixels=[bytes(128)]
    ids={}
    def handles(seq):
        result=[]
        for tile in seq:
            if tile not in ids:
                ids[tile]=len(pixels)
                pixels.append(tile)
            result.append(ids[tile])
        return result
    items=[]
    for item in inputs:
        items.append(dict(question=handles(item["question"]),page=[dict(label=handles(r["label"])) for r in item["page"]]))
    bank=torch.from_numpy(np.unpackbits(np.frombuffer(b''.join(pixels),np.uint8)).reshape(-1,32,32)).to(device)
    q,labels,_=layout(items,device)
    choices=reader(bank,q,labels).argmax(-1).tolist()
    results=[]
    for item,actions in zip(inputs,choices,strict=True):
        page=item["page"]
        rows=[]
        stopped=False
        for k in actions:
            if k>=len(page):
                stopped=True
                break
            if len(rows)==4:
                break
            rows.append(k)
        ordered=controller.reorder(item["question"],page,rows)
        assert sorted(ordered)==sorted(rows)
        both={}
        for name,indices in (("varied",rows),("ordered",ordered)):
            bits=[]
            for j,k in enumerate(indices):
                if j: bits.extend([bytes(128)]*2)
                bits.extend(page[k]["value"])
            both[name]=(strip(bits),len(bits),stopped,indices)
        results.append(both)
    return results
