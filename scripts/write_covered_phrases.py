"""Translate a pictured English or Chinese sentence by consuming source phrases."""
import argparse
from pathlib import Path
import numpy as np
from ilm.visual_lm.covered_phrases import PhraseInventory,character_fscore
from ilm.visual_lm.optical_lexicon import StripInventory,expand_table
from ilm.visual_lm.portable_writer import read_image,save_grid
from scripts.prepare_covered_phrases import MODEL,PUB,base


class CoveredWriter:
    def __init__(self,t):
        if t not in (0,1):
            raise ValueError("Expected English or Chinese as the output language")
        self.t=t
        lang=("en","zh")[t]
        tables=base.read(base.TABLE)["strip"]
        inv=base.read(base.BASE/"inventories.json")
        self.palette=np.load(base.BASE/"palette.npy")
        self.by_image={p.tobytes():i for i,p in enumerate(self.palette) if i>=2}
        self.input=StripInventory(tables[1-t],inv[1-t])
        self.input_table=tables[1-t]
        self.output=tables[t]
        path=MODEL/f"{lang}.npz"
        if base.sha(path)!=base.read(MODEL/"fit.json")["hashes"][path.name]:
            raise ValueError("Changed bilingual phrase table")
        lex=base.MODEL/lang/"probabilities.npz"
        assert base.sha(lex)==base.read(lex.parent/"fit.json")["weights"]
        with np.load(lex) as z:
            backoff=expand_table(z["ids"],z["mass"],z["tail"],z["prior"])
        self.inventory=PhraseInventory(path,len(self.output),backoff)

    def source(self,image):
        if image.dtype!=np.uint8 or image.ndim!=2 or image.shape[1]!=128 or not 1<=len(image)<=96:
            raise ValueError("Expected 1–96 aligned 32-pixel cells")
        try:
            cells=[self.by_image[p.tobytes()] for p in image]
        except KeyError as exc:
            raise ValueError("Source image contains a glyph outside the trained print repertoire") from exc
        strips=self.input.source_strips(cells)[1:]
        if [k for s in strips for k in self.input_table[s]]!=cells:
            raise ValueError("The bilingual source-strip inventory does not cover this input")
        return strips

    def translate(self,image,params=None):
        source=self.source(image)
        if params is None:
            params=base.read(MODEL/"calibration.json")["chosen"][self.t]
        output,path,score=self.inventory.decode(source,*params)
        cells=[g for q in output for g in self.output[q]]
        return self.palette[cells],path,score


def calibrate():
    dev=base.read(base.WORK/"dev.json")
    settings=[(.25,.25),(.25,1.),(.5,1.),(.5,2.),(.75,2.),(.75,3.)]
    scores=[]
    chosen=[]
    for t in (0,1):
        engine=CoveredWriter(t)
        total=np.zeros(len(settings))
        for r in dev:
            source=engine.source(engine.palette[r["cells"][1-t]])
            options=engine.inventory.lattice(source)
            for j,(beta,reward) in enumerate(settings):
                out,_,_=engine.inventory.decode(source,beta,reward,lattice=options)
                ids=[c for k in out for c in engine.output[k]]
                total[j]+=character_fscore(r["cells"][t],ids)
        scores.append((total/len(dev)).tolist())
        chosen.append(settings[int(total.argmax())])
    base.dump(MODEL/"calibration.json",dict(settings=settings,mean_glyph_f2=scores,chosen=chosen,
              dev=base.sha(base.WORK/"dev.json"),samples=len(dev),fit=base.sha(MODEL/"fit.json")))
    print(chosen,scores,flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    group=cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",type=Path)
    group.add_argument("--text")
    group.add_argument("--calibrate",action="store_true")
    cli.add_argument("--cells",type=int)
    cli.add_argument("--to",choices=["en","zh"])
    cli.add_argument("--output",type=Path)
    args=cli.parse_args()
    if args.calibrate:
        calibrate()
        return
    if not args.to or not args.output:
        cli.error("A translation requires --to and --output")
    if args.image:
        if args.cells is None:
            cli.error("Supply the occupied source cell count")
        image=read_image(args.image,args.cells)
    else:
        face=base.Ink()
        ids=face.encode(args.text)
        if face.bad:
            cli.error("Source print face lacks a requested glyph")
        image=face.array()[ids]
    pixels,path,_=CoveredWriter(int(args.to=="zh")).translate(image)
    save_grid(pixels.tobytes(),args.output)
    print(f"Covered {sum(c.end-c.start for c in path)} source strips and emitted {len(pixels)} cells.")


if __name__=="__main__":
    main()
