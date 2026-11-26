"""Locate the pictured participants of a marked event and emit their actual source ink."""
from pathlib import Path
import argparse
import torch
from safetensors.torch import load_file
from ilm.visual_lm.role_localization import RoleLocator
from scripts.locate_pictured_roles import PicturedLocator, render_sentence
from scripts.prepare_visual_referents import MODEL, read, sha
from scripts.read_natural_roles import load_strip, strip_png


class ReferentReader(PicturedLocator):
    def __init__(self, arm="target"):
        if arm not in ("local","target"):
            raise ValueError("Choose local or target")
        path = MODEL/arm/"weights.safetensors"
        if sha(path)!=read(MODEL/arm/"fit.json")["weights"]:
            raise ValueError("Reader weights differ from fit")
        self.arm = arm
        self.model = RoleLocator().eval()
        self.model.load_state_dict(load_file(str(path)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    group = cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--image",type=Path)
    cli.add_argument("--cells",type=int)
    cli.add_argument("--verb",nargs=2,type=int,required=True,metavar=("START","END"))
    cli.add_argument("--actor",type=Path,required=True)
    cli.add_argument("--undergoer",type=Path,required=True)
    args = cli.parse_args()
    if args.image and args.cells is None:
        cli.error("A source image requires the occupied cell count")
    if args.text is not None and not 1<=len(args.text)<=224:
        cli.error("Supply 1 to 224 glyph cells")
    torch.set_num_threads(2)
    ink = render_sentence(args.text) if args.image is None else load_strip(args.image.read_bytes(),args.cells)
    for path,span in zip((args.actor,args.undergoer),ReferentReader().locate(ink,args.verb),strict=True):
        path.write_bytes(strip_png(ink[slice(*span)]))


if __name__=="__main__":
    main()
