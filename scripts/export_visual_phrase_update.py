"""Package the trained visual phrase parameters alongside the existing editor renderer."""
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from scripts.compare_edit_diversity import PARENT_HASH
from scripts.infer_visual_phrase import MODELS
from scripts.train_visual_phrases import OUT, RUN, load, require, sha, tiles


def export():
    base = MODELS/'editor.safetensors'
    source = RUN/'source/writer.safetensors'
    dest = MODELS/'phrase_update.safetensors'
    require(not dest.exists() and sha(base)==PARENT_HASH and sha(source)==load(RUN/'source/fit.json')['checkpoint_sha256'],
            'Existing export or incorrect source weights')
    state = load_file(str(source))
    original = load_file(str(base))
    unchanged = [k for k in state if k=='landmarks' or k.startswith('motor.')]
    require(all(torch.equal(state[k],original[k]) for k in unchanged),'Renderer differs from base')
    weights = {k:v for k,v in state.items() if k not in unchanged}
    prompt = load(OUT/'preparation.json')['question_tiles'][0][0]
    weights['prompt_ink'] = torch.from_numpy(np.unpackbits(tiles()[prompt],axis=1)).bool()
    save_file(weights,str(dest),metadata=dict(format='ilm-visual-phrase-v1',parent_sha256=PARENT_HASH,
              trained_source_sha256=sha(source)))


if __name__=='__main__':
    export()
