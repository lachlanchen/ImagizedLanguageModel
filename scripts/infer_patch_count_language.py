"""Read an aligned Chinese text image and write a continuation as a PNG."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ilm.visual_lm.patch_count_language import PatchCountLanguage
from scripts.infer_continuous_image_language import image_cells, text_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image',type=Path,help='Black ink on white in row-major 32-by-32 cells')
    group.add_argument('--text',help='Render this prompt with the training font')
    parser.add_argument('--input-glyphs',type=int,help='Count of used cells in a padded pictured prompt')
    parser.add_argument('--maximum-new',type=int,default=128)
    parser.add_argument('--sample-seed',type=int,help='Sample the fitted distribution reproducibly')
    parser.add_argument('--output',required=True,type=Path,help='New output directory')
    args=parser.parse_args()
    if args.text is not None and args.input_glyphs is not None:
        parser.error('--input-glyphs applies to a pictured prompt')
    ink=(image_cells(args.image,args.input_glyphs) if args.image is not None else text_cells(args.text))[:,0]
    model=PatchCountLanguage.load(args.model)
    generator=np.random.default_rng(args.sample_seed) if args.sample_seed is not None else None
    answer,stopped=model.generate(ink,maximum_new=args.maximum_new,generator=generator)
    columns=min(32,max(1,len(answer)))
    rows=max(1,(len(answer)+columns-1)//columns)
    tiles=np.zeros((rows*columns,32,32),dtype=np.uint8)
    tiles[:len(answer)]=answer
    grid=tiles.reshape(rows,columns,32,32).transpose(0,2,1,3).reshape(rows*32,columns*32)
    args.output.mkdir(parents=True,exist_ok=False)
    Image.fromarray(255-255*grid).save(args.output/'answer.png')
    np.savez_compressed(args.output/'pixels.npz',ink=answer)
    with args.model.open('rb') as stream:
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(1<<20),b''):
            digest.update(block)
    report=dict(prompt_cells=len(ink),output_cells=len(answer),stopped=stopped,maximum_new=args.maximum_new,
                sample_seed=args.sample_seed,checkpoint_sha256=digest.hexdigest(),
                prompt_sha256=hashlib.sha256(ink.tobytes()).hexdigest(),output_ink_sha256=hashlib.sha256(answer.tobytes()).hexdigest())
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
