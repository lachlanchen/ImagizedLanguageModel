"""Decode complete-line training images without loading source text or fonts."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class CompleteLineImageData:
    def __init__(self,path):
        self.path=Path(path)
        self.report=json.loads((self.path/"report.json").read_text())
        if not self.report["training_only"] or not self.report["all_retained_rows_lossless"]:
            raise ValueError("expected the complete-source training image cache")
        self.names=sorted(self.report["files_sha256"])
        if len(self.names)!=sum(row["counts"]["accepted"] for row in self.report["summaries"].values()):
            raise ValueError("image cache count differs")
        self._allowed=set(self.names)

    def __len__(self):return len(self.names)

    @lru_cache(maxsize=32)
    def get(self,name):
        if name not in self._allowed:raise ValueError("image is outside the pinned training cache")
        path=self.path/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.report["files_sha256"][name]:
            raise ValueError("training image changed")
        row=torch.load(path,map_location="cpu",weights_only=False)
        width,boundary=int(row["width"]),int(row["prompt_blocks"])
        if width%64 or not 1<=boundary<width//64-2:
            raise ValueError("invalid physical image bounds")
        bits=np.unpackbits(row["bits"].numpy(),count=32*width)
        pixels=torch.from_numpy(bits.copy()).reshape(1,1,32,width).float().mul(2).sub(1)
        if not bool((pixels[...,-128:]==1).all()):raise ValueError("termination images differ")
        # No record name, category, source string, codepoint or answer position
        # is returned to the model. The boundary defines supervised loss only.
        return pixels,boundary

    def __getitem__(self,index):return self.get(self.names[index])
