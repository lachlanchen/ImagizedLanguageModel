"""Does the source-path writer retain its actual freely written answers in a second print face?"""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np
import torch

from scripts.evaluate_pictured_path_translation import ARCHIVE, PUB, ROOT, RUN, PicturedPathWriter, dump, read, sha, requests, grid_bits, save_ink
from scripts.evaluate_print_translation import FONT, normalize_print, render_face
from ilm.visual_lm.portable_writer import save_grid


def main():
    torch.set_num_threads(2)
    receipt = PUB/"transfer.json"
    archive = ARCHIVE.with_name("pictured_path_serif.zip")
    if receipt.exists():
        assert read(receipt)["archive"] == sha(archive)
        return
    assert sha(ARCHIVE) == read(PUB/"generation.json")["archive"]
    face = render_face(FONT,2)
    rows = requests()
    scores = [{} for _ in rows]
    counts = Counter()
    with ZipFile(ARCHIVE) as source, ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED) as z:
        original = json.loads(source.read("records.json"))
        serif = []
        for i,r in enumerate(rows):
            f = RUN/"serif-input.png"
            save_grid(b"".join(face.tile(c) for c in r["source"]),f)
            z.write(f,f"{i:04d}/source.png")
            serif.append(grid_bits(f,len(r["input"])))
        normalized = normalize_print(serif,"cpu")
        for i,r in enumerate(rows):
            reference = grid_bits(BytesIO(source.read(f"{i:04d}/source.png")),len(r["input"]))
            counts["cells"]+=len(reference)
            for k,g in (("direct",serif),("normalized",normalized)):
                counts[k+"_matching_cells"]+=int(np.all(g[i]==reference,axis=(1,2)).sum())
        for t,lang in enumerate(("en","zh")):
            model=PicturedPathWriter(lang,"cpu")
            group=[i for i,r in enumerate(rows) if r["t"]==t]
            for k,images in (("direct",serif),("normalized",normalized)):
                for j in range(0,len(group),24):
                    ix=group[j:j+24]
                    for i,(bits,stop,edges) in zip(ix,model.write([images[n] for n in ix]),strict=True):
                        raw=BytesIO()
                        save_ink(bits,raw,image_format="PNG")
                        data=raw.getvalue()
                        z.writestr(f"{i:04d}/{k}.png",data)
                        baseline=original[i]["answers"]["path"]
                        same=(data,stop,edges)==(source.read(f"{i:04d}/path.png"),baseline["stop"],baseline["boundaries"])
                        counts[k+"_matching_answers"]+=same
                        scores[i][k]=dict(sha256=sha256(data).hexdigest(),identical=same,stop=stop,boundaries=edges)
        z.writestr("records.json",json.dumps(scores))
    archive.with_suffix(".tmp").replace(archive)
    dump(receipt,dict(archive=sha(archive),canonical=sha(ARCHIVE),font=sha(FONT),counts=counts,
                      protocol=sha(ROOT/"references/pictured_path_transfer_protocol.md")))
    print(counts,flush=True)


if __name__=="__main__":
    main()
