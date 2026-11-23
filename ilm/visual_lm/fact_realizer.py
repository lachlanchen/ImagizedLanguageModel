"""Read factual fields and verbalize them by autoregressively emitting image cells."""
import json
from pathlib import Path
import numpy as np
import torch

from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder


class FactRealizer:
    def __init__(self, writer, device="cpu"):
        folder=Path(writer)
        cues=json.loads((folder/"cues.json").read_text())
        self.roles=[bytes.fromhex(v) for v in cues["roles"]]
        self.commands={k:bytes.fromhex(v) for k,v in cues["commands"].items()}
        self.writer=PreparedGlyphResponder(folder,device)

    def answer_fields(self,fields,order):
        instruction=self.commands[order]
        def tensor(raw):
            return torch.from_numpy(np.frombuffer(raw,np.uint8).copy().reshape(-1,128))
        pairs=[(tensor(subject+bytes(128)+self.roles[role]+bytes(128)+obj),tensor(instruction))
               for subject,role,obj in fields]
        return [(y.numpy().tobytes(),stop) for y,stop in self.writer.answer_many(pairs)]
