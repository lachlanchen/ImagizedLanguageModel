"""Verify the 48-way receipt using the public search interface and the archived PNGs only."""
from io import BytesIO
import json
from zipfile import ZipFile
import numpy as np
from scripts.fit_optical_lexicon import PUB,ROOT,read,dump,sha
from scripts.search_optical_passages import PixelPassageSearch
from scripts.audit_full_image_roles import image_cells


def main():
    archive=ROOT/"publication/ilm-nature/evidence/optical_lexicon.zip"
    assert sha(archive)==read(PUB/"generation.json")["archive"]
    scores=[]
    with ZipFile(archive) as z:
        data=json.loads(z.read("records.json"))
        imgs=[image_cells(z.read(f"{i:04d}/source.png"),len(r["source"])) for i,r in enumerate(data)]
        for t,lang in enumerate(("en","zh")):
            lookup=PixelPassageSearch(lang)
            targets=imgs[1-t::2]
            candidates=imgs[t::2]
            for i,img in enumerate(targets):
                ranked=np.array(lookup.score(img,candidates))
                rival=np.delete(ranked,i).max()
                scores.append(dict(t=t,i=i,strict=bool(ranked[i]>rival)))
    total=sum(r["strict"] for r in scores)
    assert total==read(PUB/"retrieval.json")["splits"]["all_sources/lexical"]["strictly_first"]
    dump(PUB/"pixel-search.json",dict(strict=total,n=len(scores),choices=48,archive=sha(archive),records=scores))
    print(total,len(scores))


if __name__=="__main__":
    main()
