"""Return an answer image from separate aligned passage and question PNGs."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits


def input_cells(path, count):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('L'))
    h, w = pixels.shape
    if h % 32 or w % 32 or not np.isin(pixels, (0, 255)).all():
        raise ValueError('Expected a binary page of aligned 32 by 32 cells')
    grid = (pixels < 128).reshape(h//32,32,w//32,32).transpose(0,2,1,3).reshape(-1,1024)
    if count is not None:
        if not 1 <= count <= len(grid) or grid[count:].any():
            raise ValueError('Invalid cell count or nonempty trailing padding')
        grid = grid[:count]
    return np.packbits(grid, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path, help='Trained local reader.joblib')
    for name in ('passage','question'):
        parser.add_argument('--'+name, required=True, type=Path)
        parser.add_argument('--'+name+'-cells', type=int, help='Exclude known trailing page padding')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(2):
        reader = joblib.load(args.model)
        passage = input_cells(args.passage, args.passage_cells)
        question = input_cells(args.question, args.question_cells)
        left, right = reader.bounds(passage, question)
    answer = np.unpackbits(passage[left:right], axis=1).reshape(-1,32,32)
    cols = min(32,len(answer))
    grid = np.zeros(((len(answer)+cols-1)//cols*cols,32,32), np.uint8)
    grid[:len(answer)] = answer
    pixels = grid.reshape(-1,cols,32,32).transpose(0,2,1,3).reshape(-1,cols*32)
    Image.fromarray(255-255*pixels).save(args.output)
    print(f'Wrote {right-left} cells from passage offsets [{left}, {right}) to {args.output}')


if __name__=='__main__':
    main()
