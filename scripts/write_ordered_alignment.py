"""Read a printed sentence image and freely translate it with learned source movement."""
import argparse
from pathlib import Path
import torch
from ilm.visual_lm.ordered_alignment import AlignedPrefix,emission_laws,movement
from ilm.visual_lm.optical_lexicon import StripPrefix
from ilm.visual_lm.portable_writer import read_image, save_grid
from scripts.prepare_ordered_alignment import MODEL, base
from scripts.write_optical_lexicon import LexicalWriter


class OrderedWriter(LexicalWriter):
    def __init__(self,device="cpu"):
        super().__init__(device)
        self.weights=[]
        for lang in ("en","zh"):
            fit=base.read(MODEL/lang/"fit.json")
            if fit["inputs"]["emissions"]!=base.sha(base.MODEL/lang/"probabilities.npz"):
                raise ValueError("The frozen emission weights differ from the alignment fit")
            self.weights.append(fit["mixing"])

    def laws(self,sources,t):
        table,prior=self.models[t]
        states=[]
        for src in sources:
            ids=self.strips[1-t].source_strips(src)[1:]
            emission=emission_laws(table,ids)
            aligned=AlignedPrefix(self.strips[t],emission,movement(len(emission),self.weights[t]))
            states.append((aligned,StripPrefix(self.strips[t],prior)))
        return states


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    group=cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--image",type=Path)
    cli.add_argument("--cells",type=int)
    cli.add_argument("--to",required=True,choices=["en","zh"])
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    torch.set_num_threads(2)
    if args.image:
        if args.cells is None:
            cli.error("Provide the occupied cell count of the source PNG")
        image=read_image(args.image,args.cells)
    else:
        face=base.Ink()
        ids=face.encode(args.text)
        if face.bad:
            cli.error("Missing source glyph in the input typeface")
        image=face.array()[ids]
    (result,stop),=OrderedWriter().translate([image],int(args.to=="zh"),.5)
    save_grid(result.tobytes(),args.output)
    print(f"Emitted {len(result)} glyphs. Natural stop {stop}.")


if __name__=="__main__":
    main()
