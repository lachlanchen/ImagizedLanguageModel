"""Read a pictured question and passage in aligned cells and write a Serif answer image."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from ilm.visual_lm.grounded_detail_writer import GroundedDetailWriter, answer
from scripts.infer_continuous_image_language import image_cells, text_cells
from scripts.train_visual_extractive_reader import save_page


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reader',required=True,type=Path,help='Trusted trained passage reader from this project')
    parser.add_argument('--writer',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    for name in ('passage','question'):
        group=parser.add_mutually_exclusive_group(required=True)
        group.add_argument(f'--{name}-image',type=Path,help='Row-major 32-by-32 black-on-white image cells')
        group.add_argument(f'--{name}-text')
        parser.add_argument(f'--{name}-cells',type=int,help='Used cells in the pictured input')
    args=parser.parse_args()
    inputs=[]
    for name in ('passage','question'):
        image=getattr(args,name+'_image')
        count=getattr(args,name+'_cells')
        if count is not None and image is None:
            parser.error(f'--{name}-cells applies only to a pictured input')
        ink=image_cells(image,count) if image is not None else text_cells(getattr(args,name+'_text'))
        inputs.append(np.packbits(ink.reshape(-1,1024),axis=1))
    reader=joblib.load(args.reader)
    writer=GroundedDetailWriter.load(args.writer)
    with threadpool_limits(limits=2):
        output,(left,right)=answer(reader,writer,*inputs)
    args.output.mkdir(parents=True,exist_ok=False)
    save_page(args.output/'answer.png',output)
    np.savez_compressed(args.output/'answer.npz',packed=output)
    receipt=dict(source_start=left,source_end_exclusive=right,output_cells=len(output),
                 passage_cells=len(inputs[0]),question_cells=len(inputs[1]),
                 output_sha256=hashlib.sha256(output.tobytes()).hexdigest())
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt)


if __name__=='__main__':
    main()
