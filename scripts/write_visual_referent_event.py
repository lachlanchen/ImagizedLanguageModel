"""Write the marked affirmative English event in Chinese from an aligned print image."""
import argparse
from pathlib import Path
import numpy as np
import torch

from ilm.visual_lm.portable_writer import save_grid
from scripts.read_visual_referents import ReferentReader, load_strip, render_sentence
from scripts.prepare_region_writer import Ink
from scripts.write_ordered_translation import restore, stopped
from scripts.train_paired_ink import BASE, material


class EventWriter:
    def __init__(self, device="cpu"):
        self.device = device
        self.reader = ReferentReader()
        self.writer = restore("book-rehearsal",device)
        self.features, inventory, _ = material(device)
        self.vocabulary = inventory[1]
        self.palette = np.load(BASE/"palette.npy")
        face = Ink(self.palette)
        self.alphabet = {}
        for c in map(chr,range(32,127)):
            cell, = face.encode(c)
            if cell < len(self.palette):
                self.alphabet[render_sentence(c)[0].tobytes()] = cell
        self.blank,self.period = face.encode(" .")

    @torch.inference_mode()
    def write(self, source, verb):
        actor, undergoer = self.reader.locate(source,verb)
        event = []
        for j,span in enumerate((actor,verb,undergoer)):
            if j:
                event.append(self.blank)
            for image in source[slice(*span)]:
                cell = self.alphabet.get(image.tobytes())
                if cell is None:
                    raise ValueError("The English event writer expects the supplied 32-pixel DejaVu print face")
                event.append(cell)
        event.append(self.period)
        if len(event)>96:
            raise ValueError("The selected event exceeds the 96-cell writing window")
        seq = self.writer.generate(self.features,torch.tensor([event],device=self.device),self.vocabulary,1)
        ids,stop = stopped(seq[0].cpu())
        return self.palette[ids].tobytes(),stop,(actor,undergoer)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    group = cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--image",type=Path)
    cli.add_argument("--cells",type=int)
    cli.add_argument("--verb",required=True,nargs=2,type=int,metavar=("START","END"))
    cli.add_argument("--output",required=True,type=Path)
    opt = cli.parse_args()
    if opt.image is not None and opt.cells is None:
        cli.error("Supply the occupied cell count for the input PNG")
    if opt.text is not None and not 1<=len(opt.text)<=224:
        cli.error("Supply 1 to 224 occupied glyph cells")
    torch.set_num_threads(2)
    ink = render_sentence(opt.text) if opt.image is None else load_strip(opt.image.read_bytes(),opt.cells)
    raw,stop,spans = EventWriter().write(ink,opt.verb)
    save_grid(raw,opt.output)
    print(f"Participant intervals {spans}. Emitted {len(raw)//128} cells. Stopped {stop}.")


if __name__=="__main__":
    main()
