"""Write a title image from a pictured or typed article using the fitted transducer."""
import argparse
from pathlib import Path
import json

import numpy as np
import torch

from ilm.visual_lm.title_transducer import TitleTransducer
from scripts.infer_continuous_image_language import image_cells, text_cells
from scripts.prepare_visual_title_composition import QUESTION
from scripts.train_visual_extractive_reader import save_page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=Path, help='Black-on-white row-major grid of 32-pixel square cells')
    group.add_argument('--text')
    parser.add_argument('--cells', type=int, help='Number of used cells in a padded input image')
    parser.add_argument('--question-image', type=Path, help='Single-row title instruction in the same format')
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.cells is not None and args.image is None:
        parser.error('--cells requires an image')
    passage = image_cells(args.image,args.cells) if args.image else text_cells(args.text)
    question = image_cells(args.question_image,None) if args.question_image else text_cells(QUESTION)
    torch.set_num_threads(2)
    model = TitleTransducer().eval()
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    packed = lambda a: np.packbits(a.reshape(-1,1024), axis=1)
    result, actions, stopped = model.write(packed(passage),packed(question))
    output = np.asarray(result, dtype=np.uint8).reshape(-1,128)
    args.output.mkdir(exist_ok=False,parents=True)
    if len(output):
        save_page(args.output/'title.png',output)
    (args.output/'title.bin').write_bytes(output.tobytes())
    receipt=dict(cells=len(output),drawn=sum(actions),stopped=stopped)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
