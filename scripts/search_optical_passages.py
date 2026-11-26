"""Rank pictured passages in one language using a pictured query in the other."""
import argparse
import json
from pathlib import Path
import numpy as np
from ilm.visual_lm.optical_lexicon import StripInventory,expand_table
from ilm.visual_lm.portable_writer import read_image
from scripts.fit_optical_lexicon import BASE,MODEL,TABLE,read,sha


class PixelPassageSearch:
    def __init__(self,query_language):
        t={"en":0,"zh":1}[query_language]
        self.t=t
        folder=MODEL/query_language
        if sha(folder/"probabilities.npz")!=read(folder/"fit.json")["weights"]:
            raise ValueError("The lexical table differs from its training receipt")
        with np.load(folder/"probabilities.npz") as z:
            self.table=expand_table(z["ids"],z["mass"],z["tail"],z["prior"])
        tables=read(TABLE)["strip"]
        inventory=read(BASE/"inventories.json")
        self.strip=[StripInventory(z,inv) for z,inv in zip(tables,inventory,strict=True)]
        names=read(BASE/"readback.json")
        self.content=[]
        for lang,table in enumerate(tables):
            def has_script(span):
                chars="".join(names.get(str(c),"") for c in span)
                return any("a"<=c.lower()<="z" if lang==0 else "\u3400"<=c<="\u9fff" for c in chars)
            self.content.append({i for i,span in enumerate(table) if has_script(span)})
        self.by_image={tile.tobytes():i for i,tile in enumerate(np.load(BASE/"palette.npy")) if i>=2}

    def pieces(self,image,t):
        if image.dtype!=np.uint8 or image.ndim!=2 or image.shape[1]!=128 or not len(image):
            raise ValueError("Provide an ordered nonempty grid of packed 32-pixel glyphs")
        cells=[self.by_image.get(x.tobytes(),-1) for x in image]
        return [k for k in self.strip[t].source_strips(cells) if k in self.content[t]]

    def score(self,query,candidates):
        target=self.pieces(query,self.t)
        if not target:
            raise ValueError("No script-bearing query strip in the trained print inventory")
        result=[]
        for image in candidates:
            src=self.pieces(image,1-self.t)
            mass=self.table[np.ix_([0,*src],target)].mean(0)
            result.append(float(np.log(mass).mean()))
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query",required=True,type=Path)
    parser.add_argument("--cells",required=True,type=int)
    parser.add_argument("--language",required=True,choices=("en","zh"))
    parser.add_argument("--candidates",required=True,type=Path,help="JSON array of image paths and occupied cell counts")
    args=parser.parse_args()
    candidates=read(args.candidates)
    if not candidates:
        parser.error("Supply at least one candidate")
    search=PixelPassageSearch(args.language)
    images=[read_image(args.candidates.parent/r["image"],r["cells"]) for r in candidates]
    scores=search.score(read_image(args.query,args.cells),images)
    ordered=sorted((dict(**r,score=s) for r,s in zip(candidates,scores,strict=True)),key=lambda r:-r["score"])
    print(json.dumps(ordered,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
