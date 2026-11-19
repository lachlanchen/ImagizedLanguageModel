"""An image-only question responder with a learned copy-or-compose decision."""

from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch
from torch.nn.utils.rnn import pad_sequence

from ilm.visual_lm.composed_answer_writer import descriptors
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.grounded_glyph_decoder import GroundedGlyphDecoder
from ilm.visual_lm.interval_writer import IntervalWriter
from ilm.visual_lm.scoped_question import ScopedQuestion


class VisualPageResponder:
    @torch.inference_mode()
    def __init__(self, reader_dir, writer_dir, device):
        path=Path(writer_dir)
        self.device=device
        self.reader=FontInvariantReader.restore(reader_dir,device)
        self.writer=GroundedGlyphDecoder()
        self.writer.question=ScopedQuestion(ordered=True)
        self.writer.load_state_dict(load_file(str(path/"final.safetensors")))
        self.writer.to(device).eval()
        self.selector=IntervalWriter().to(device).eval()
        self.selector.load_state_dict(load_file(str(path/"interval.safetensors")))
        original=list(map(bytes,np.load(path/"memory.npy",allow_pickle=False)))
        self.tiles=[bytes(128),*original]
        self.index={cell:i for i,cell in enumerate(original,1)}
        self.mem=torch.arange(1,len(original)+1,device=device)
        self.ink=self.embed(self.tiles)
        self.keys=self.writer.write_key(self.ink[self.mem])

    def embed(self,cells):
        packed=np.frombuffer(b"".join(cells),np.uint8).reshape(-1,128)
        bits=torch.from_numpy(np.unpackbits(packed,axis=1)).to(self.device)
        return self.writer.ink(descriptors(bits))

    @torch.inference_mode()
    def answer_batch(self, pairs):
        if not pairs: return []
        additions={}
        def addresses(raw):
            if not raw or len(raw)%128:
                raise ValueError("Expected a nonempty stream of 32 by 32 glyph images")
            indices=[]
            for i in range(0,len(raw),128):
                cell=raw[i:i+128]
                k=self.index.get(cell)
                if k is None:
                    k=additions.setdefault(cell,len(self.tiles)+len(additions))
                indices.append(k)
            return torch.tensor(indices,device=self.device)
        p,q=(pad_sequence([addresses(s) for s in stream],batch_first=True) for stream in zip(*pairs,strict=True))
        ink=torch.cat((self.ink,self.embed(additions))) if additions else self.ink
        action,_=self.selector(self.writer.encode(ink,p,q),p)
        copy=action>=0
        result=[None]*len(pairs)
        indices=torch.where(~copy)[0]
        if len(indices):
            generated,ended=self.writer.generate(ink,p[indices],q[indices],self.keys,self.mem,limit=64)
            cells=self.tiles+list(additions)
            for i,row,stop in zip(indices.tolist(),generated.tolist(),ended.tolist(),strict=True):
                result[i]=(b"".join(cells[k] for k in row if k),stop,"compose")
        for i in torch.where(copy)[0].tolist():
            image,_=self.reader.answer(*pairs[i])
            result[i]=(image,True,"read")
        return result
