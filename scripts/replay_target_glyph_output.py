"""Reproduce the target-supervision study from the original PNGs and model weights."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from ilm.visual_lm.field_to_glyph import ROOT, restore
from ilm.visual_lm.target_glyph_teaching import load_writer
from scripts.check_field_writer_roundtrip import cells
from scripts.write_neural_glyphs import grid_bits, save_ink

MODES = ("start","likelihood","guided","exchange")
ARCHIVE = ROOT/"publication/ilm-nature/evidence/target_glyph_answers.zip"
PUB = ROOT/"references/evidence/target_glyph_teaching_v1"


@torch.inference_mode()
def write_condition(mode, device, grids, directions, sources, batch_size):
    net = restore("prediction", device)[0] if mode=="start" else load_writer("guided" if mode=="exchange" else mode,device)
    for t in (0,1):
        order = [i for i,d in enumerate(directions) if d==t]
        for start in range(0,len(order),batch_size):
            ix = order[start:start+batch_size]
            images = [grids[sources[i]] for i in ix]
            sizes = [len(x) for x in images]
            x = np.zeros((len(ix),max(sizes),32,32),dtype=bool)
            for j,g in enumerate(images):
                x[j,:len(g)] = g
            inks, counts, stop = net.write(torch.from_numpy(x).to(device),torch.tensor(sizes,device=device),t)
            ink_cpu = inks.cpu().numpy()
            for j,i in enumerate(ix):
                count = int(counts[j])
                blob = BytesIO()
                save_ink(ink_cpu[j,:count],blob,image_format="PNG")
                yield i, blob.getvalue(), count, bool(stop[j])


def main():
    torch.set_num_threads(2)
    rec = json.loads((PUB/"evaluation.json").read_text())
    if rec["archive"]!=sha256(ARCHIVE.read_bytes()).hexdigest():
        raise ValueError("Archive binding")
    exact, differences, reproduced = {}, [], Counter()
    with ZipFile(ARCHIVE) as z:
        rows = json.loads(z.read("request-outcomes.json"))
        grids = [grid_bits(BytesIO(z.read(f"{i:04d}/input.png")),len(r["source"])) for i,r in enumerate(rows)]
        directions = [("en","zh").index(r["target_language"]) for r in rows]
        if len(rows)!=192 or any(rows[i^1]["source"]!=r["reference"] for i,r in enumerate(rows)):
            raise ValueError("Incomplete bilingual archive")
        for m in MODES:
            exact[m] = []
            sources = [r["answers"][m]["source_index"] for r in rows]
            for i,png,n,stop in write_condition(m,"cpu",grids,directions,sources,12):
                saved = z.read(f"{i:04d}/{m}.png")
                r = rows[i]["answers"][m]
                if sha256(saved).hexdigest()!=r["png"]:
                    raise ValueError(f"Changed archived PNG {i}/{m}")
                if png==saved and (n,stop)==(r["cells"],r["stop"]):
                    reproduced[m]+=1
                else:
                    differences.append(dict(i=i,mode=m,cells=n,stopped=stop,png=sha256(png).hexdigest()))
                if r["stop"] and np.array_equal(cells(saved,r["cells"]),grids[i^1]):
                    exact[m].append(i)
    report = dict(requests=len(rows),reproduced=reproduced,differences=differences,exact_reference=exact,
                  archive=rec["archive"],program=sha256(Path(__file__).read_bytes()).hexdigest())
    (PUB/"replay.json").write_text(json.dumps(report,indent=2)+"\n")
    print("CPU replay", reproduced, "exact references",exact,flush=True)
    if differences:
        raise SystemExit("CPU and GPU outputs differ")


if __name__=="__main__":
    main()
