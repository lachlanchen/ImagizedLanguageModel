"""Answer a pictured question by returning a sequence of original source images."""

import argparse
from pathlib import Path

import torch

from ilm.visual_lm.continuous_page_reader import ContinuousPageReader
from ilm.visual_lm.portable_writer import read_image, save_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    for field in ("passage","question"):
        parser.add_argument("--"+field,type=Path,required=True)
        parser.add_argument("--"+field+"-cells",type=int)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    args=parser.parse_args()
    torch.set_num_threads(2)
    reader=ContinuousPageReader.restore(args.checkpoint,args.device)
    p=read_image(args.passage,args.passage_cells).tobytes()
    q=read_image(args.question,args.question_cells).tobytes()
    result,(a,b)=reader.answer(p,q)
    save_grid(result,args.output)
    print(f"Returned {b-a} source glyphs from position {a}.")


if __name__=="__main__":
    main()
