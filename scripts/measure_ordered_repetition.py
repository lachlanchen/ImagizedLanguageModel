"""Exploratory paired study of repeated content in the original free answer images."""
from hashlib import sha256
import json
from zipfile import ZipFile
import numpy as np
from scripts.audit_full_image_roles import image_cells
from scripts.evaluate_ordered_alignment import ARCHIVE,ARMS
from scripts.prepare_ordered_alignment import PUB,base
from scripts.search_optical_passages import PixelPassageSearch


def repeats(strips):
    windows=list(zip(strips,strips[1:],strips[2:]))
    return len(windows)-len(set(windows))


def main():
    palette=np.load(base.BASE/"palette.npy")
    ink=base.Ink(palette)
    search=[PixelPassageSearch(s) for s in ("en","zh")]
    measured=[]
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read("records.json"))
        for i,r in enumerate(rows):
            if r["group"]!="book":
                continue
            t=r["t"]
            engine=search[t]
            ids=ink.encode(r["ref"])
            streams={"reference":engine.pieces(ink.array()[ids],t)}
            for m in ARMS:
                a=r["answers"][m]
                raw=z.read(f"{i:04d}/{m}.png")
                assert sha256(raw).hexdigest()==a["png"]
                streams[m]=engine.pieces(image_cells(raw,a["cells"]),t) if a["cells"] else []
            shortest=min(map(len,streams.values()))
            measured.append({m:dict(length=len(seq),repeats=repeats(seq),equal_prefix=repeats(seq[:shortest]))
                                  for m,seq in streams.items()})
    counts={}
    for m in (*ARMS,"reference"):
        counts[m]=dict(any_repeat=sum(r[m]["repeats"]>0 for r in measured),
                  equal_prefix_any=sum(r[m]["equal_prefix"]>0 for r in measured),
                  repeated_windows=sum(r[m]["repeats"] for r in measured),
                  total_windows=sum(max(r[m]["length"]-2,0) for r in measured))
    base.dump(PUB/"repetition.json",dict(archive=base.sha(ARCHIVE),counts=counts,observations=measured,
                  definition="Exact ordered triples of script-bearing strips; each duplicate after the first counts. Equal prefixes truncate the reference and all three answers at the per-request minimum strip length."))
    print(counts)


if __name__=="__main__":
    main()
