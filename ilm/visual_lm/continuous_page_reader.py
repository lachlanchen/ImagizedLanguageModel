"""A pictured-span reader with continuous pixel input and source-image output."""

from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertForQuestionAnswering


def visual_features(pixels):
    ink = pixels.to(torch.float32).reshape(-1, 1, 32, 32)
    scales = (ink.flatten(1), F.avg_pool2d(ink,2).flatten(1), F.avg_pool2d(ink,4).flatten(1))
    return torch.cat([F.normalize(x,dim=1)*w for x,w in zip(scales,(.8,.5,.33),strict=True)],dim=1)


class ContinuousPageReader(nn.Module):
    def __init__(self, config, projection, regression, markers):
        super().__init__()
        # A vocabulary of size zero keeps the contextual encoder API, with no lexical lookup.
        config.vocab_size = 0
        config.pad_token_id = None
        self.encoder = BertForQuestionAnswering(config)
        self.encoder.bert.embeddings.word_embeddings = None
        self.register_buffer("projection", projection)
        self.register_buffer("regression", regression)
        self.register_buffer("markers", markers)

    def hidden_features(self, pixels):
        return F.relu(visual_features(pixels) @ self.projection) * (2/self.projection.shape[1])**.5

    def embed(self, raw):
        data = np.frombuffer(raw, np.uint8).reshape(-1,128)
        device = self.markers.device
        chunks = []
        for start in range(0,len(data),128):
            bits = torch.from_numpy(np.unpackbits(data[start:start+128], axis=1)).to(device)
            chunks.append(self.hidden_features(bits) @ self.regression)
        return torch.cat(chunks)

    @torch.inference_mode()
    def answer(self, passage, question, max_answer=64):
        if len(passage)%128 or len(question)%128 or not passage or not question:
            raise ValueError("Expected nonempty sequences of packed 32 by 32 images")
        p,q = self.embed(passage),self.embed(question)
        stop = self.encoder.config.max_position_embeddings - len(q) - 3
        if stop < max_answer:
            raise ValueError("Question exceeds the encoder context budget")
        best = -float("inf")
        chosen = None
        for offset in range(0,len(p),stop-max_answer+1):
            segment = p[offset:offset+stop]
            seq = torch.cat((self.markers[0:1],q,self.markers[1:2],segment,self.markers[1:2]))
            ids = torch.zeros((1,len(seq)),dtype=torch.long,device=seq.device)
            begin = len(q)+2
            ids[:,begin:] = 1
            out = self.encoder(inputs_embeds=seq[None],token_type_ids=ids)
            left = out.start_logits[0,begin:begin+len(segment)]
            right = out.end_logits[0,begin:begin+len(segment)]
            scores = left[:,None]+F.pad(right,(0,max_answer-1),value=-torch.inf).unfold(0,max_answer,1)
            score,index = scores.flatten().max(0)
            if float(score)>best:
                start,length = divmod(int(index),max_answer)
                chosen=(offset+start,offset+start+length+1)
                best=float(score)
            if offset+len(segment)==len(p):
                break
        a,b=chosen
        return passage[a*128:b*128], (a,b)

    @classmethod
    def restore(cls, folder, device="cpu"):
        folder=Path(folder)
        config=BertConfig.from_json_file(str(folder/"config.json"))
        tables=load_file(str(folder/"vision.safetensors"))
        reader=cls(config,**tables)
        missing,extra=reader.encoder.load_state_dict(load_file(str(folder/"context.safetensors")),strict=False)
        if missing or extra:
            raise ValueError(f"Context weight mismatch: {missing}, {extra}")
        return reader.to(device).eval()
