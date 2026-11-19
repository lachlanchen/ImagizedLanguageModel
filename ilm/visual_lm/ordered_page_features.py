"""Frozen contextual features for a span-and-glyph image writer."""
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def cells(raw):
    if not raw or len(raw) % 128:
        raise ValueError("Expected a nonempty stream of packed 32 by 32 images")
    return [raw[i:i+128] for i in range(0,len(raw),128)]


@torch.no_grad()
def contextualize(reader, pairs, batch_size=16):
    device = reader.markers.device
    indexed = {}
    def indices(raw):
        return [indexed.setdefault(tile,len(indexed)) for tile in cells(raw)]
    inputs = [(indices(p),indices(q)) for p,q in pairs]
    if any(len(p)+len(q)+3 > reader.encoder.config.max_position_embeddings for p,q in inputs):
        raise ValueError("Passage and question exceed the contextual reader's 512-cell window")
    tiles = list(indexed)
    features = torch.cat([reader.embed(b"".join(tiles[i:i+128])) for i in range(0,len(tiles),128)])
    order = sorted(range(len(inputs)),key=lambda i:sum(map(len,inputs[i])))
    outputs = [None]*len(inputs)
    for first in range(0,len(order),batch_size):
        chosen = order[first:first+batch_size]
        sequences, roles, extras = [],[],[]
        for i in chosen:
            p,q = inputs[i]
            sequences.append(torch.cat((reader.markers[:1],features[q],reader.markers[1:2],
                                        features[p],reader.markers[1:2])))
            roles.append(torch.tensor([0]*(len(q)+2)+[1]*(len(p)+1),device=device))
            pos = {x:(j+1)/len(q) for j,x in enumerate(q)}
            extras.append([(float(x in pos),pos.get(x,0.),(j+1)/len(p)) for j,x in enumerate(p)])
        seq = pad_sequence(sequences,batch_first=True)
        mask = torch.arange(seq.shape[1],device=device)[None] < torch.tensor(
            [len(x) for x in sequences],device=device)[:,None]
        h = reader.encoder.bert(inputs_embeds=seq,attention_mask=mask,
                                token_type_ids=pad_sequence(roles,batch_first=True)).last_hidden_state
        prior = reader.encoder.qa_outputs(h)
        for j,i in enumerate(chosen):
            p,q = inputs[i]
            span = slice(len(q)+2,len(q)+2+len(p))
            local = torch.cat((h[j,span],prior[j,span],torch.tensor(extras[j],device=device)),dim=-1)
            outputs[i] = (h[j,0].half().cpu(), local.half().cpu())
    return outputs
