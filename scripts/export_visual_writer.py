"""Export only changed physical writer tensors and bind them to the portable base."""
import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from scripts.fit_visual_writer_distillation import OUT, ROOT, WRITER

def main():
    base = ROOT/'models/pictured_idiom_zh/reader_writer.safetensors'
    destination = base.with_name('distilled_writer.safetensors')
    if destination.exists():
        raise FileExistsError(destination)
    original = load_file(str(base))
    trained = torch.load(OUT/'distilled_targets/weights.pt', map_location='cpu', weights_only=True)
    if set(original) != set(trained):
        raise ValueError('Architecture differs')
    changed = {k:v for k,v in trained.items() if not torch.equal(v,original[k])}
    if set(changed) != {k for k in trained if k.startswith(WRITER)}:
        raise ValueError('Unexpected changed tensors')
    save_file(changed,str(destination),metadata={'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),
              'format':'ilm-physical-writer-v1'})
    merged = {**original,**load_file(str(destination))}
    assert all(torch.equal(merged[k],v) for k,v in trained.items())
    print(len(changed),destination.stat().st_size,hashlib.sha256(destination.read_bytes()).hexdigest())

if __name__=='__main__':
    main()
